"""Hands-free, streaming full-duplex server (the low-latency production path).

Protocol over ``/ws`` (all audio 16 kHz mono PCM16):

* client -> server binary: ``uint32 seq`` followed by exactly 320 samples
  (20 ms). ``seq`` starts at 0, so sample index = ``seq * 320``.
* server -> client binary: ``uint32 turn``, ``uint32 chunk`` then PCM16 of one
  synthesized clause.
* server -> client JSON events: ``endpoint`` (with the acoustic ``end_sample``
  of the user's speech), ``transcript``, ``duck`` / ``resume`` / ``stop`` (with
  the barge-in ``onset_sample``), ``reply_done`` (server stage timings).
* client -> server JSON events: ``playback_started`` (client-clock
  ``t_first_audio_ms`` from the captured end-of-speech frame to the scheduled
  start of the first clause), ``duck_ack`` / ``stop_ack`` (client-clock
  latency from the captured onset frame), ``playback_ended``.

The microphone is processed continuously: endpointing while the user speaks,
barge-in control while the assistant responds. Turn telemetry can be appended
to a JSONL file for automated loopback evaluation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from thesis_s2s.runtime.bargein_live import BargeInController, InterruptClassifier
from thesis_s2s.runtime.streaming import (
    ClauseChunker,
    OpenAIStreamingResponder,
    StreamingResponder,
    run_streamed_turn,
)
from thesis_s2s.runtime.vad import SAMPLE_RATE, Endpointer, FrameScorer, default_scorer

FRAME = 320
PRE_ROLL_SAMPLES = int(0.2 * SAMPLE_RATE)
MAX_TURN_SECONDS = 30.0


@dataclass
class LiveComponents:
    asr: Callable[[np.ndarray], tuple[str, str | None]]
    responder: StreamingResponder | OpenAIStreamingResponder
    tts: Callable[[str], np.ndarray]
    classifier: InterruptClassifier | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    scorer_factory: Callable[[], FrameScorer] = default_scorer


@dataclass
class LiveConfig:
    hangover_ms: float = 256.0
    history_turns: int = 2
    telemetry_path: Path | None = None
    confirm_speech_ms: float = 480.0
    classifier_threshold: float = 0.7
    speculative_ms: float | None = 160.0
    first_min_words: int = 2
    first_max_words: int = 4
    max_words: int = 14


class _Session:
    """Per-connection state machine (listening <-> responding)."""

    def __init__(self, components: LiveComponents, config: LiveConfig) -> None:
        self.components = components
        self.config = config
        self.endpointer = Endpointer(
            scorer=components.scorer_factory(),
            hangover_ms=config.hangover_ms,
            tentative_ms=config.speculative_ms,
        )
        self.bargein = BargeInController(
            scorer=components.scorer_factory(),
            classifier=components.classifier,
            confirm_speech_ms=config.confirm_speech_ms,
            classifier_threshold=config.classifier_threshold,
        )
        self.recent: np.ndarray = np.zeros(0, dtype=np.float32)
        self.recent_start = 0
        self.utterance_start: int | None = None
        self.turn = 0
        self.responding = False
        self.cancel = threading.Event()
        self.history: list[tuple[str, str]] = []
        self.records: dict[int, dict[str, Any]] = {}
        self.gate = threading.Event()
        self.gate_lock = threading.Lock()
        self.held: list[bytes] = []
        self.speculative_turn: int | None = None
        self.speculative_end: int | None = None

    def remember(self, pcm: np.ndarray, start: int) -> None:
        if not self.recent.size:
            self.recent_start = start
        self.recent = np.concatenate([self.recent, pcm])
        limit = int(MAX_TURN_SECONDS * SAMPLE_RATE)
        if self.recent.size > limit:
            drop = self.recent.size - limit
            self.recent = self.recent[drop:]
            self.recent_start += drop

    def audio_between(self, start: int, end: int) -> np.ndarray:
        a = max(0, start - self.recent_start)
        b = max(a, end - self.recent_start)
        return self.recent[a:b].copy()


def _pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def build_live_app(components: LiveComponents, config: LiveConfig | None = None) -> FastAPI:
    config = config or LiveConfig()
    app = FastAPI(title="Persian full-duplex streaming assistant", version="1.0.0")
    telemetry_lock = threading.Lock()

    def write_telemetry(record: dict[str, Any]) -> None:
        if config.telemetry_path is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with telemetry_lock, config.telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "mode": "live_streaming", "sample_rate": SAMPLE_RATE, **components.identity}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return LIVE_HTML

    @app.websocket("/ws")
    async def ws_live(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        session = _Session(components, config)
        generation: asyncio.Future[Any] | None = None

        async def send_json(payload: dict[str, Any]) -> None:
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 - client went away
                pass

        def start_turn(
            utterance: np.ndarray,
            end_sample: int,
            decided_sample: int,
            *,
            speculative: bool = False,
        ) -> None:
            nonlocal generation
            session.turn += 1
            turn = session.turn
            session.cancel = threading.Event()
            cancel = session.cancel
            with session.gate_lock:
                session.gate = threading.Event()
                session.held = []
                if not speculative:
                    session.gate.set()
            gate, gate_lock = session.gate, session.gate_lock
            session.speculative_turn = turn if speculative else None
            session.speculative_end = end_sample if speculative else None
            if not speculative:
                session.responding = True
                session.bargein.reset()
            record: dict[str, Any] = {
                "turn": turn,
                "end_sample": end_sample,
                "endpoint_decided_sample": decided_sample,
                "speculative_start": speculative,
                "utterance_seconds": round(utterance.size / SAMPLE_RATE, 3),
                "server_endpoint_wall": time.time(),
            }
            session.records[turn] = record

            def emit(audio: np.ndarray, index: int) -> None:
                payload = struct.pack("<II", turn, index) + _pcm16(audio)
                with gate_lock:
                    if not gate.is_set():
                        session.held.append(payload)
                        return
                asyncio.run_coroutine_threadsafe(ws.send_bytes(payload), loop)

            def run() -> Any:
                return run_streamed_turn(
                    utterance,
                    asr=components.asr,
                    responder=components.responder,
                    tts=components.tts,
                    emit_audio=emit,
                    history=list(session.history),
                    cancel=cancel,
                    chunker=ClauseChunker(
                        first_min_words=config.first_min_words,
                        first_max_words=config.first_max_words,
                        max_words=config.max_words,
                    ),
                )

            async def finish() -> None:
                result = await loop.run_in_executor(None, run)
                record.update(
                    {
                        "transcript": result.transcript,
                        "reply_text": result.reply_text,
                        "clauses": len(result.clauses),
                        "server_timings_ms": result.timings.as_ms(),
                        "asr_error": result.asr_error,
                        "responder_error": result.responder_error,
                        "cancelled": result.cancelled,
                        "reply_samples": result.audio_samples,
                    }
                )
                if record.get("speculation_cancelled"):
                    return
                done = {
                    "event": "reply_done",
                    "turn": turn,
                    "transcript": result.transcript,
                    "reply_text": result.reply_text,
                    "server_timings_ms": result.timings.as_ms(),
                    "cancelled": result.cancelled,
                    "clauses": len(result.clauses),
                }
                if session.speculative_turn == turn:
                    # Not yet confirmed: deliver after the endpoint is confirmed.
                    record["pending_reply_done"] = done
                    return
                await deliver_reply(done)

            generation = asyncio.ensure_future(finish())
            if not speculative:
                asyncio.ensure_future(
                    send_json({"event": "endpoint", "turn": turn, "end_sample": end_sample})
                )

        async def deliver_reply(done: dict[str, Any]) -> None:
            if done["transcript"] and done["reply_text"]:
                session.history.append((done["transcript"], done["reply_text"]))
                session.history = session.history[-config.history_turns :]
            await send_json(done)

        async def confirm_speculative(end_sample: int) -> None:
            """Open the gate of the speculative turn and release held audio."""

            turn = session.speculative_turn
            session.speculative_turn = None
            session.responding = True
            session.bargein.reset()
            record = session.records.get(turn or -1, {})
            record["speculation_confirmed"] = True
            await send_json({"event": "endpoint", "turn": turn, "end_sample": end_sample})
            # Drain held clauses in order; open the gate only once nothing is held,
            # so a clause emitted during the drain cannot overtake earlier ones.
            while True:
                with session.gate_lock:
                    held, session.held = session.held, []
                    if not held:
                        session.gate.set()
                        break
                for payload in held:
                    await ws.send_bytes(payload)
            pending = record.pop("pending_reply_done", None)
            if pending is not None:
                await deliver_reply(pending)

        def cancel_speculative() -> None:
            turn = session.speculative_turn
            if turn is None:
                return
            session.speculative_turn = None
            session.cancel.set()
            session.records.setdefault(turn, {})["speculation_cancelled"] = True
            with session.gate_lock:
                session.held = []

        def finish_record(turn: int, **fields: Any) -> None:
            record = session.records.get(turn)
            if record is None:
                return
            record.update(fields)
            if record.get("closed"):
                return
            if fields.get("close"):
                record["closed"] = True
                write_telemetry(record)

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                raw = message.get("bytes")
                if raw is not None:
                    if len(raw) != 4 + 2 * FRAME:
                        await send_json({"event": "error", "message": "invalid frame size"})
                        continue
                    (seq,) = struct.unpack("<I", raw[:4])
                    start = seq * FRAME
                    pcm = np.frombuffer(raw[4:], dtype="<i2").astype(np.float32) / 32768.0
                    session.remember(pcm, start)
                    if not session.responding:
                        for event in session.endpointer.push(pcm, sample_offset=start):
                            if event.kind == "speech_start":
                                session.utterance_start = max(0, event.sample - PRE_ROLL_SAMPLES)
                            elif event.kind == "speech_pause" and session.utterance_start is not None:
                                utterance = session.audio_between(
                                    session.utterance_start, event.sample
                                )
                                start_turn(
                                    utterance, event.sample, event.decided_at_sample, speculative=True
                                )
                            elif event.kind == "speech_resume":
                                cancel_speculative()
                            elif event.kind == "speech_end" and session.utterance_start is not None:
                                if (
                                    session.speculative_turn is not None
                                    and session.speculative_end == event.sample
                                ):
                                    session.utterance_start = None
                                    await confirm_speculative(event.sample)
                                    continue
                                cancel_speculative()
                                utterance = session.audio_between(
                                    session.utterance_start, event.sample
                                )
                                session.utterance_start = None
                                start_turn(utterance, event.sample, event.decided_at_sample)
                    else:
                        for barge in session.bargein.push(pcm, sample_offset=start):
                            turn = session.turn
                            await send_json(
                                {
                                    "event": barge.kind,
                                    "turn": turn,
                                    "onset_sample": barge.onset_sample,
                                    "decided_sample": barge.decided_at_sample,
                                    "probability": barge.probability,
                                }
                            )
                            record = session.records.get(turn, {})
                            record.setdefault("bargein_events", []).append(
                                {
                                    "kind": barge.kind,
                                    "onset_sample": barge.onset_sample,
                                    "decided_sample": barge.decided_at_sample,
                                    "probability": barge.probability,
                                }
                            )
                            if barge.kind == "stop":
                                session.cancel.set()
                                session.responding = False
                                session.endpointer.reset()
                                pre_roll_start = max(0, barge.onset_sample - PRE_ROLL_SAMPLES)
                                pre_roll = session.audio_between(pre_roll_start, start + FRAME)
                                session.utterance_start = pre_roll_start
                                session.endpointer.push(pre_roll, sample_offset=pre_roll_start)
                elif message.get("text"):
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await send_json({"event": "error", "message": "invalid JSON"})
                        continue
                    event = payload.get("event")
                    turn = int(payload.get("turn") or 0)
                    if event == "playback_started":
                        finish_record(
                            turn,
                            client_t_first_audio_ms=payload.get("t_first_audio_ms"),
                            client_t_first_audible_ms=payload.get("t_first_audible_ms"),
                        )
                    elif event in {"duck_ack", "stop_ack"}:
                        key = "client_t_duck_ms" if event == "duck_ack" else "client_t_stop_ms"
                        finish_record(turn, **{key: payload.get("latency_ms")})
                        if event == "stop_ack":
                            finish_record(turn, close=True, outcome="interrupted")
                    elif event == "playback_ended":
                        if turn == session.turn:
                            session.responding = False
                            session.endpointer.reset()
                        finish_record(turn, close=True, outcome="completed")
                    elif event == "hello":
                        await send_json({"event": "ready", **components.identity})
        except WebSocketDisconnect:
            pass
        finally:
            session.cancel.set()
            if generation is not None and not generation.done():
                generation.cancel()
            for turn, record in session.records.items():
                if not record.get("closed"):
                    finish_record(turn, close=True, outcome="disconnected")

    return app


LIVE_HTML = r"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>دستیار صوتی فارسی</title>
<style>
body{font-family:Tahoma,sans-serif;background:#111;color:#f5f5f5;margin:0;padding:24px}
h1{font-size:2.2rem}p{font-size:1.3rem;line-height:1.8}
button{font-size:1.7rem;padding:18px 30px;border:0;border-radius:14px;background:#2e7d32;color:#fff}
#log{white-space:pre-wrap;background:#222;padding:16px;min-height:160px;font-size:1.15rem}
</style></head>
<body>
<h1>دستیار صوتی فارسی</h1>
<p>دکمه را یک بار بزنید و طبیعی صحبت کنید. لازم نیست دوباره دکمه بزنید؛ هر وقت خواستید وسط
پاسخ دستیار حرف بزنید تا ساکت شود.</p>
<button id="start">شروع گفت‌وگو</button>
<div id="log">آماده.</div>
<script>
const logEl=document.getElementById('log');
const log=(t)=>{logEl.textContent=t+'\n'+logEl.textContent.slice(0,2000);};
const ctx=new (window.AudioContext||window.webkitAudioContext)();
const out=ctx.createGain(); out.connect(ctx.destination);
let ws, seq=0, carry=new Float32Array(0);
const frameTimes=new Map();
let nextStart=0, currentTurn=0, firstReported=new Set(), endSample=new Map();
const sources=new Set();
function sampleTime(sample){
  const s=Math.floor(sample/320), t=frameTimes.get(s);
  if(t===undefined) return null;
  return t-((319-(sample%320))/16000)*1000;
}
function resampleTo16k(x, rate){
  if(rate===16000) return x;
  const n=Math.floor(x.length*16000/rate), y=new Float32Array(n);
  for(let i=0;i<n;i++){const p=i*rate/16000, l=Math.floor(p), r=Math.min(x.length-1,l+1), f=p-l; y[i]=x[l]*(1-f)+x[r]*f;}
  return y;
}
function sendFrames(mono){
  let buf=new Float32Array(carry.length+mono.length); buf.set(carry); buf.set(mono, carry.length);
  let off=0;
  while(buf.length-off>=320){
    const msg=new ArrayBuffer(4+640), dv=new DataView(msg);
    dv.setUint32(0, seq, true);
    for(let i=0;i<320;i++){dv.setInt16(4+2*i, Math.max(-32768,Math.min(32767,Math.round(buf[off+i]*32767))), true);}
    frameTimes.set(seq, performance.now());
    if(frameTimes.size>6000) frameTimes.delete(seq-6000);
    if(ws&&ws.readyState===1) ws.send(msg);
    seq++; off+=320;
  }
  carry=buf.slice(off);
}
function stopAll(){ for(const s of sources){try{s.stop();}catch(_){}} sources.clear(); nextStart=0; }
function playChunk(turn, index, pcm){
  if(turn<currentTurn) return;
  currentTurn=turn;
  const f=new Float32Array(pcm.length); for(let i=0;i<pcm.length;i++) f[i]=pcm[i]/32768;
  const b=ctx.createBuffer(1,f.length,16000); b.getChannelData(0).set(f);
  const src=ctx.createBufferSource(); src.buffer=b; src.connect(out);
  const when=Math.max(ctx.currentTime+0.01, nextStart); nextStart=when+b.duration;
  src.start(when); sources.add(src);
  src.onended=()=>{sources.delete(src); if(sources.size===0 && ws.readyState===1){ws.send(JSON.stringify({event:'playback_ended', turn}));}};
  if(!firstReported.has(turn) && endSample.has(turn)){
    firstReported.add(turn);
    const startPerf=performance.now()+(when-ctx.currentTime)*1000;
    const ref=sampleTime(endSample.get(turn));
    const audible=startPerf+((ctx.outputLatency||0)+(ctx.baseLatency||0))*1000;
    ws.send(JSON.stringify({event:'playback_started', turn,
      t_first_audio_ms: ref===null?null:startPerf-ref, t_first_audible_ms: ref===null?null:audible-ref}));
  }
}
async function start(){
  await ctx.resume();
  const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:false,autoGainControl:false}});
  const src=ctx.createMediaStreamSource(stream);
  const proc=ctx.createScriptProcessor(1024,1,1);
  const mute=ctx.createGain(); mute.gain.value=0;
  src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
  proc.onaudioprocess=(e)=>sendFrames(resampleTo16k(e.inputBuffer.getChannelData(0), ctx.sampleRate));
  ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.binaryType='arraybuffer';
  ws.onopen=()=>ws.send(JSON.stringify({event:'hello'}));
  ws.onmessage=(ev)=>{
    if(typeof ev.data!=='string'){const dv=new DataView(ev.data); playChunk(dv.getUint32(0,true), dv.getUint32(4,true), new Int16Array(ev.data.slice(8))); return;}
    const m=JSON.parse(ev.data);
    if(m.event==='endpoint'){endSample.set(m.turn, m.end_sample); log('در حال پاسخ...');}
    else if(m.event==='duck'){out.gain.setTargetAtTime(0.2, ctx.currentTime, 0.01); const r=sampleTime(m.onset_sample); ws.send(JSON.stringify({event:'duck_ack', turn:m.turn, latency_ms:r===null?null:performance.now()-r}));}
    else if(m.event==='resume'){out.gain.setTargetAtTime(1.0, ctx.currentTime, 0.02);}
    else if(m.event==='stop'){stopAll(); out.gain.setValueAtTime(1.0, ctx.currentTime); currentTurn=m.turn+1; const r=sampleTime(m.onset_sample); ws.send(JSON.stringify({event:'stop_ack', turn:m.turn, latency_ms:r===null?null:performance.now()-r})); log('ساکت شدم؛ بفرمایید.');}
    else if(m.event==='reply_done'){log('شما: '+(m.transcript||'')+'\nدستیار: '+(m.reply_text||''));}
    window.__events=(window.__events||[]); window.__events.push(m);
  };
}
document.getElementById('start').onclick=()=>{start(); document.getElementById('start').disabled=true; log('گوش می‌دهم...');};
if(new URLSearchParams(location.search).get('autostart')==='1') start();
</script></body></html>
"""


def _low_latency_piper(model: Path | None) -> Any:
    """Load Piper with a bounded ONNX Runtime thread pool (lower, steadier latency)."""

    import os

    import onnxruntime as ort
    from piper import PiperVoice
    from piper.config import PiperConfig

    if model is None or not model.is_file():
        raise RuntimeError("the pinned Persian Piper voice is required")
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("PIPER_INTRA_OP_THREADS", "8"))
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    config = PiperConfig.from_dict(
        json.loads(Path(f"{model}.json").read_text(encoding="utf-8"))
    )
    return PiperVoice(session=session, config=config)


def default_live_components(
    *,
    system_prompt: str | None = None,
    max_new_tokens: int = 64,
    history_turns: int = 2,
) -> LiveComponents:
    """Production components: HTTP ASR service, streamed local LLM, strict Piper TTS.

    The ASR service URL comes from ``ASR_API_URL`` (see ``scripts/serve_asr_nemo.py``).
    Piper failures raise instead of silently falling back to formant speech.
    """

    import os

    from thesis_s2s.runtime.cascade import (
        QWEN4B_PROMPT_V2,
        TextResponder,
        _asr_transcript,
        _wav_bytes,
        asr_http,
    )
    from thesis_s2s.runtime.tts import os_piper_model

    prompt = system_prompt or QWEN4B_PROMPT_V2
    endpoint = os.environ.get("TEXT_LLM_ENDPOINT", "").strip()
    responder: StreamingResponder | OpenAIStreamingResponder
    if endpoint:
        responder = OpenAIStreamingResponder(
            endpoint=endpoint,
            model=os.environ.get("TEXT_LLM_SERVED_NAME", "qwen3-4b-instruct-2507"),
            system_prompt=prompt,
            max_new_tokens=max_new_tokens,
            history_turns=history_turns,
        )
        model_name = os.environ.get("TEXT_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
        model_revision = os.environ.get("TEXT_LLM_REVISION", "cdbee75f17c01a7cc42f958dc650907174af0554")
        backend = "openai_compatible_stream"
    else:
        base = TextResponder()
        if base.model is None or base.tokenizer is None:
            raise RuntimeError(f"responder failed to load: {base.initialization_error}")
        responder = StreamingResponder(
            base.model,
            base.tokenizer,
            system_prompt=prompt,
            device=base.device,
            max_new_tokens=max_new_tokens,
            history_turns=history_turns,
        )
        model_name = base.model_name
        model_revision = base.model_revision or "unspecified"
        backend = "transformers"

    def asr(audio: np.ndarray) -> tuple[str, str | None]:
        result = asr_http(_wav_bytes(audio))
        error = str(result["error"]) if result.get("error") else None
        return _asr_transcript(result), error

    voice = _low_latency_piper(os_piper_model())

    def tts(text: str) -> np.ndarray:
        from piper.config import SynthesisConfig

        from thesis_s2s.audio import resample

        chunks = list(
            voice.synthesize(text, syn_config=SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0))
        )
        if not chunks:
            raise RuntimeError("Piper synthesis failed")
        audio = np.concatenate([np.asarray(c.audio_float_array, dtype=np.float32) for c in chunks])
        audio = resample(audio, int(chunks[0].sample_rate), SAMPLE_RATE)
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("Piper synthesis failed")
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    for _ in range(2):  # warm the LLM kernels and the Piper session
        list(responder.stream("سلام"))
        tts("سلام، بفرمایید.")
    piper = os_piper_model()
    identity = {
        "asr_backend": "http",
        "responder_model": model_name,
        "responder_revision": model_revision,
        "responder_backend": backend,
        "system_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tts_model": piper.name if piper is not None else None,
        "persian_only_decoding": True,
        "streaming": True,
    }
    return LiveComponents(asr=asr, responder=responder, tts=tts, identity=identity)
