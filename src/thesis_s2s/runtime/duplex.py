"""Full-duplex control runtime with an honest modular Persian cascade.

Recording kit (`--record`) writes 16 kHz near-mic WAV + JSONL of T_first_audio /
T_barge_in / stop events. Study ratings POST to `/study/rating`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from thesis_s2s import SAMPLE_RATE, T_FIRST_AUDIO_P50_MS
from thesis_s2s.bargein.detector import BargeinDetector, EnergyVadBaseline
from thesis_s2s.bargein.realtime import PlaybackController
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.config import project_root
from thesis_s2s.metrics import gpu_inventory
from thesis_s2s.model.llama_omni2 import checkpoint_runtime_status
from thesis_s2s.runtime.cascade import QWEN4B_MODEL, QWEN4B_REVISION, CascadeTalker
from thesis_s2s.runtime.session_log import (
    PROMPTS,
    SessionMeta,
    SessionStore,
    _append_jsonl_atomic,
)
from thesis_s2s.runtime.tts import FormantTalker, piper_runtime_ready


@dataclass
class TurnLog:
    t_first_audio_ms: float | None = None
    server_generation_ms: float | None = None
    t_barge_in_ms: float | None = None
    stopped: bool = False
    talker: str = ""
    tts_backend: str = ""


class StudyRating(BaseModel):
    session_id: str | None = None
    speaker_id: str | None = None
    age_bin: str | None = None
    naturalness: int | None = Field(default=None, ge=1, le=5)
    latency: int | None = Field(default=None, ge=1, le=5)
    interrupt_success: int | None = Field(default=None, ge=1, le=5)
    satisfaction: int | None = Field(default=None, ge=1, le=5)
    would_talk_again: bool | None = None
    elderly_notes: str = Field(default="", max_length=2000)
    detector: str | None = None
    consent: bool = False


class DummyTalker(FormantTalker):
    """Tests-only talker: formant first packet so T_first_audio stays under 500 ms."""

    def __init__(self, reply_text: str = "سلام"):
        super().__init__(reply_text=reply_text)
        from thesis_s2s.runtime.tts import first_packet, formant_synthesize

        audio = formant_synthesize(self.reply_text)
        self.backend = "formant"
        self._cached_first = first_packet(audio)

    def first_chunk(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        from thesis_s2s.runtime.tts import first_packet, formant_synthesize

        if text is None:
            if self._cached_first is None:
                raise RuntimeError("dummy talker first packet is unavailable")
            return self._cached_first
        self.backend = "formant"
        return first_packet(formant_synthesize(text))


def default_talker():
    """Return the only implemented conversational runtime: the modular cascade."""

    return CascadeTalker()


class DuplexSession:
    def __init__(self, detector: BargeinDetector | EnergyVadBaseline, talker=None):
        self.controller = PlaybackController(detector)
        self.talker = talker or default_talker()
        self.log = TurnLog()
        self.mic_buffer: np.ndarray = np.zeros(0, dtype=np.float32)

    def on_user_end(self, user_audio: np.ndarray, text: str | None = None) -> np.ndarray:
        t0 = time.perf_counter()
        reply_audio = getattr(self.talker, "reply_audio", None)
        full_reply = getattr(self.talker, "full_reply", None)
        if callable(reply_audio):
            chunk = reply_audio(user_audio, text)
        else:
            chunk = (
                full_reply(text)
                if callable(full_reply)
                else self.talker.first_chunk(user_audio, text)
            )
        self.log.server_generation_ms = 1000.0 * (time.perf_counter() - t0)
        self.log.t_first_audio_ms = None
        self.log.talker = type(self.talker).__name__
        self.log.tts_backend = getattr(self.talker, "backend", "")
        self.controller.start_playback()
        return chunk

    def on_mic_while_playing(self, chunk: np.ndarray, interrupt_onset: bool = False) -> bool:
        stopped = self.controller.on_mic_chunk(chunk, interrupt_onset=interrupt_onset)
        if stopped:
            self.log.stopped = True
            self.log.t_barge_in_ms = self.controller.t_barge_in_ms()
        return stopped


ELDERLY_HTML = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>گفت‌وگوی صوتی فارسی</title>
  <style>
    body { font-family: Tahoma, sans-serif; background:#111; color:#f5f5f5; margin:0; padding:24px; }
    h1 { font-size: 2.2rem; }
    p, label { font-size: 1.25rem; line-height:1.7; }
    button { font-size: 1.6rem; padding: 18px 28px; margin: 8px; border:0; border-radius:12px; cursor:pointer; }
    #talk { background:#2e7d32; color:white; }
    #stop { background:#c62828; color:white; }
    #log { white-space:pre-wrap; background:#222; padding:16px; min-height:120px; font-size:1.1rem; }
    .row { margin: 8px 0; }
    input, select, textarea { font-size: 1.1rem; padding: 8px; }
  </style>
</head>
<body>
  <h1>دستیار صوتی فارسی</h1>
  <p>دکمه سبز را بزنید، حرف بزنید، و دوباره بزنید تا پاسخ بیاید. برای آزمون قطع خودکار، هنگام پاسخ ربات شروع به صحبت کنید؛ دکمه قرمز فقط توقف دستی اضطراری است.</p>
  <p><strong>سیاست داده:</strong> __RETENTION_NOTICE__</p>
  <div class="row">
    <label>نشست <input id="session" value="S001"/></label>
    <label>گوینده <input id="speaker" value="P01"/></label>
    <label>سن
      <select id="age">
        <option value="under_60">زیر ۶۰</option>
        <option value="60plus">۶۰+</option>
      </select>
    </label>
  </div>
  <div class="row">
    <label>برگه
      <select id="prompt">
        <option value="warmup_time">گرم‌کردن</option>
        <option value="interrupt_story">قطع داستان</option>
        <option value="backchannel">آها/بله</option>
        <option value="noise">نویز</option>
        <option value="free">آزاد</option>
      </select>
    </label>
    <label>برچسب
      <select id="label">
        <option value="none">none</option>
        <option value="interrupt">interrupt</option>
        <option value="backchannel">backchannel</option>
        <option value="noise">noise</option>
      </select>
    </label>
  </div>
  <div class="row">
    <label><input id="consent" type="checkbox"/> رضایت‌نامه و سیاست دادهٔ بالا را خوانده‌ام و با پردازش این نشست موافقم.</label>
  </div>
  <button id="talk">صحبت کن</button>
  <button id="stop">توقف دستی</button>
  <div id="log">آماده.</div>
  <h2>ارزیابی پایانی</h2>
  <div class="row">
    <label>طبیعی بودن <select id="naturalness"></select></label>
    <label>سرعت پاسخ <select id="latency"></select></label>
    <label>موفقیت قطع <select id="interrupt_success"></select></label>
    <label>رضایت کلی <select id="satisfaction"></select></label>
  </div>
  <div class="row"><label><input id="again" type="checkbox"/> دوباره با این سامانه صحبت می‌کنم.</label></div>
  <div class="row"><label>یادداشت <textarea id="notes" maxlength="2000"></textarea></label></div>
  <button id="rate">ثبت ارزیابی</button>


  <script>
    const q = new URLSearchParams(location.search);
    const log = (t) => { document.getElementById('log').textContent = t; };
    ['session','speaker','age'].forEach((id) => {
      if (q.get(id)) document.getElementById(id).value = q.get(id);
    });
    if (q.get('prompt')) document.getElementById('prompt').value = q.get('prompt');
    if (q.get('label')) document.getElementById('label').value = q.get('label');
    ['naturalness','latency','interrupt_success','satisfaction'].forEach((id) => {
      const el = document.getElementById(id);
      for (let value=1; value<=5; value++) el.add(new Option(String(value), String(value)));
    });
    const expectedLabels = {warmup_time:'none', interrupt_story:'interrupt',
      backchannel:'backchannel', noise:'noise', free:'none'};
    document.getElementById('prompt').onchange = (event) => {
      document.getElementById('label').value = expectedLabels[event.target.value] || 'none';
    };

    let ws, stream, processor, collecting=false, playing=false;
    let eosSentAt=null, interruptOnsetAt=null, clientFirstReported=false;
    const sources = new Set();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    function meta() {
      return {
        session_id: document.getElementById('session').value,
        speaker_id: document.getElementById('speaker').value,
        age_bin: document.getElementById('age').value,
        prompt_id: document.getElementById('prompt').value,
        interrupt_label: document.getElementById('label').value,
        consent: document.getElementById('consent').checked,
      };
    }
    function resample16k(input, inputRate) {
      const n = Math.max(1, Math.round(input.length * 16000 / inputRate));
      const out = new Float32Array(n);
      for (let i=0; i<n; i++) {
        const pos = i * inputRate / 16000;
        const left = Math.min(input.length-1, Math.floor(pos));
        const right = Math.min(input.length-1, left+1);
        const mix = pos-left;
        out[i] = input[left]*(1-mix) + input[right]*mix;
      }
      return out;
    }
    function stopPlayback(reason) {
      if (!playing) return;
      playing = false;
      const measured = interruptOnsetAt === null ? null : performance.now()-interruptOnsetAt;
      for (const src of sources) { try { src.stop(); } catch (_) {} }
      sources.clear();
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({event:'playback_stopped_ack', reason,
          t_barge_in_ms:measured, ...meta()}));
      }
      interruptOnsetAt = null;
      log('پخش واقعاً متوقف شد.');
    }
    async function ensureMic() {
      if (processor) return;
      stream = await navigator.mediaDevices.getUserMedia({audio:{
        channelCount:1, echoCancellation:true, noiseSuppression:false, autoGainControl:false
      }});
      const input = ctx.createMediaStreamSource(stream);
      processor = ctx.createScriptProcessor(2048, 1, 1);
      const silent = ctx.createGain(); silent.gain.value = 0;
      input.connect(processor); processor.connect(silent); silent.connect(ctx.destination);
      processor.onaudioprocess = (event) => {
        const mono = resample16k(event.inputBuffer.getChannelData(0), ctx.sampleRate);
        let sum = 0;
        for (let i=0; i<mono.length; i++) sum += mono[i]*mono[i];
        const rms = Math.sqrt(sum/mono.length);
        if (playing && rms > 0.025 && interruptOnsetAt === null) {
          interruptOnsetAt = performance.now();
        }
        if ((collecting || playing) && ws && ws.readyState === WebSocket.OPEN) {
          const pcm = new Int16Array(mono.length);
          for (let i=0; i<mono.length; i++) {
            pcm[i] = Math.max(-32768, Math.min(32767, Math.round(mono[i]*32767)));
          }
          ws.send(pcm.buffer);
        }
      };
    }
    function connect() {
      ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
      ws.binaryType = 'arraybuffer';
      ws.onopen = () => ws.send(JSON.stringify({event:'hello', ...meta()}));
      ws.onmessage = async (ev) => {
        if (typeof ev.data === 'string') {
          const msg = JSON.parse(ev.data);
          if (msg.event === 'stopped') {
            // The server has retained the rolling microphone pre-roll. Keep
            // streaming the same utterance so the user's interruption becomes
            // the next conversational turn instead of being discarded.
            collecting = true;
            stopPlayback('server_detector');
            log('پخش متوقف شد؛ ادامه دهید و برای پایان گفتار دکمه سبز را بزنید.');
          }
          else if (msg.event === 'cancelled') {
            stopPlayback('server_cancel');
            collecting = false;
            log('نوبت جاری لغو شد.');
          }
          else if (msg.event === 'error') log(msg.message);
          return;
        }
        const pcm = new Int16Array(ev.data);
        const f32 = new Float32Array(pcm.length);
        for (let i=0;i<pcm.length;i++) f32[i]=pcm[i]/32768;
        const audio = ctx.createBuffer(1, f32.length, 16000);
        audio.getChannelData(0).set(f32);
        const src = ctx.createBufferSource();
        src.buffer = audio; src.connect(ctx.destination);
        sources.add(src); playing = true;
        src.onended = () => {
          sources.delete(src);
          if (playing && sources.size === 0) {
            playing = false;
            ws.send(JSON.stringify({event:'playback_ended', ...meta()}));
          }
        };
        src.start(ctx.currentTime + 0.005);
        if (!clientFirstReported && eosSentAt !== null) {
          clientFirstReported = true;
          ws.send(JSON.stringify({event:'playback_started',
            t_first_audio_ms:performance.now()-eosSentAt+5, ...meta()}));
        }
      };
    }
    connect();
    document.getElementById('talk').onclick = async () => {
      await ctx.resume();
      await ensureMic();
      if (!collecting) {
        collecting = true;
        ws.send(JSON.stringify({event:'begin_utterance', ...meta()}));
        log('در حال گوش دادن... میکروفن هنگام پاسخ هم برای قطع خودکار فعال می‌ماند.');
      } else {
        collecting = false;
        eosSentAt = performance.now();
        clientFirstReported = false;
        interruptOnsetAt = null;
        ws.send(JSON.stringify({event:'end_of_speech', ...meta()}));
        log('در حال پاسخ...');
      }
    };
    document.getElementById('stop').onclick = () => {
      interruptOnsetAt = performance.now();
      collecting = true;
      ws.send(JSON.stringify({event:'interrupt', ...meta()}));
    };
    document.getElementById('rate').onclick = async () => {
      const payload = {
        ...meta(),
        naturalness:Number(document.getElementById('naturalness').value),
        latency:Number(document.getElementById('latency').value),
        interrupt_success:Number(document.getElementById('interrupt_success').value),
        satisfaction:Number(document.getElementById('satisfaction').value),
        would_talk_again:document.getElementById('again').checked,
        elderly_notes:document.getElementById('notes').value,
      };
      const response = await fetch('/study/rating', {
        method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(payload)
      });
      if (response.ok) log('ارزیابی ثبت شد. سپاسگزاریم.');
      else {
        const detail = await response.json();
        log('ثبت ارزیابی ناموفق بود: '+(detail.detail || response.status));
      }
    };

  </script>
</body>
</html>
"""


def build_app(
    detector: BargeinDetector | EnergyVadBaseline | None = None,
    *,
    record: bool = False,
    study: bool = False,
    retention: str | None = None,
    session_id: str = "S001",
    speaker_id: str = "P01",
    age_bin: str = "under_60",
    detector_name: str = "gbdt",
) -> FastAPI:
    app = FastAPI(title="Persian duplex S2S", version="0.3.0")
    if retention is not None and retention not in {"audio", "features", "metrics"}:
        raise ValueError("retention must be 'audio', 'features', or 'metrics'")
    if record and retention not in {None, "audio"}:
        raise ValueError(
            "--record is an audio-retention alias and conflicts with another retention mode"
        )
    effective_retention = retention or ("audio" if record else "features" if study else "metrics")
    talker = default_talker()
    det: BargeinDetector | EnergyVadBaseline
    model_path = project_root() / "results" / "bargein" / "bargein_gbdt.pkl"
    if detector is not None:
        det = detector
    elif detector_name == "energy":
        det = EnergyVadBaseline()
    elif model_path.is_file():
        det = BargeinDetector.load(model_path)
    else:
        det = EnergyVadBaseline()
    host_gpu = gpu_inventory()
    store = SessionStore() if record or study or retention is not None else None

    @app.get("/health")
    def health() -> dict:
        responder = getattr(talker, "responder", None)
        validated_cascade_ready = bool(
            isinstance(talker, CascadeTalker)
            and responder is not None
            and getattr(responder, "model", None) is not None
            and getattr(responder, "tokenizer", None) is not None
            and getattr(responder, "model_name", None) == QWEN4B_MODEL
            and getattr(responder, "model_revision", None) == QWEN4B_REVISION
            and getattr(responder, "prompt_profile", None) == "qwen4b_v2"
            and piper_runtime_ready()
        )
        return {
            "ok": True,
            "validated_cascade_ready": validated_cascade_ready,
            "sample_rate": SAMPLE_RATE,
            "t_first_audio_budget_ms": T_FIRST_AUDIO_P50_MS,
            "talker": type(talker).__name__,
            "tts_backend": getattr(talker, "backend", ""),
            "responder_backend": getattr(responder, "backend", None),
            "responder_revision": getattr(responder, "model_revision", None),
            "responder_prompt_profile": getattr(responder, "prompt_profile", None),
            "omni_checkpoint": checkpoint_runtime_status(
                project_root() / "checkpoints" / "llama_omni2_fa" / "persian_omni2.pt"
            ),
            "record": record,
            "study": study,
            "retention": effective_retention if store is not None else "disabled",
            "session_id": session_id,
        }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        notices = {
            "audio": "صدای خام و شاخص‌های زمان‌بندی ذخیره می‌شوند.",
            "features": "صدای خام ذخیره نمی‌شود؛ فقط آمار تجمیعی انرژی، زیروبمی و MFCC و شاخص‌های زمان‌بندی نگه‌داری می‌شوند.",
            "metrics": "صدای خام و ویژگی صوتی ذخیره نمی‌شود؛ فقط برچسب، امتیاز و شاخص‌های زمان‌بندی نگه‌داری می‌شوند.",
        }
        notice = notices.get(effective_retention, "هیچ داده‌ای ذخیره نمی‌شود.")
        return ELDERLY_HTML.replace("__RETENTION_NOTICE__", notice)

    @app.get("/prompts")
    def prompts() -> dict:
        return {"prompts": PROMPTS}

    @app.post("/simulate_turn")
    def simulate_turn() -> JSONResponse:
        session = DuplexSession(det, talker)
        user = _harmonic(0.8, 200, rng=np.random.default_rng(2))
        reply = session.on_user_end(user)
        interrupted = session.on_mic_while_playing(user, interrupt_onset=True)
        return JSONResponse(
            {
                "reply_samples": int(len(reply)),
                "interrupted": interrupted,
                "t_first_audio_ms": session.log.t_first_audio_ms,
                "t_barge_in_ms": session.log.t_barge_in_ms,
                "talker": session.log.talker,
                "tts_backend": session.log.tts_backend,
            }
        )

    @app.post("/study/rating")
    def study_rating(payload: StudyRating) -> dict:
        if not study:
            raise HTTPException(status_code=404, detail="study mode is disabled")
        if not payload.consent:
            raise HTTPException(status_code=403, detail="explicit consent is required")
        sid = str(payload.session_id or session_id)
        rating_store = store or SessionStore()
        rating_meta = SessionMeta(
            session_id=sid,
            speaker_id=str(payload.speaker_id or speaker_id),
            gpu_name=str(host_gpu.get("device") or "cpu"),
            vram_gb=float(host_gpu["total_gb"]) if host_gpu.get("total_gb") is not None else None,
            memory_capped=bool(host_gpu.get("memory_capped", False)),
            age_bin=str(payload.age_bin or age_bin),
            consent=True,
            retention=effective_retention,
        )
        try:
            folder = rating_store.start(rating_meta)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        path = folder / "mos.jsonl"
        row = payload.model_dump()
        row.update(
            {
                "session_id": sid,
                "speaker_id": rating_meta.speaker_id,
                "age_bin": rating_meta.age_bin,
                "detector": payload.detector or detector_name,
            }
        )
        _append_jsonl_atomic(path, row)
        return {"ok": True, "session_id": sid}

    @app.websocket("/ws")
    async def ws_duplex(ws: WebSocket) -> None:
        await ws.accept()
        session = DuplexSession(det, talker)
        buf: list[np.ndarray] = []
        collecting = False
        pending_turn: dict | None = None
        turn_meta = {
            "prompt_id": "warmup_time",
            "interrupt_label": "none",
            "session_id": session_id,
            "speaker_id": speaker_id,
            "age_bin": age_bin,
            "consent": False,
        }
        local_meta = SessionMeta(
            session_id=session_id,
            speaker_id=speaker_id,
            gpu_name=str(host_gpu.get("device") or "cpu"),
            vram_gb=float(host_gpu["total_gb"]) if host_gpu.get("total_gb") is not None else None,
            memory_capped=bool(host_gpu.get("memory_capped", False)),
            age_bin=age_bin,
            consent=False,
            retention=effective_retention,
        )

        def valid_ms(value) -> float | None:
            if isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 60_000:
                return float(value)
            return None

        def persist_pending() -> None:
            nonlocal pending_turn
            if pending_turn is None:
                return
            pending_meta = pending_turn["meta"]
            if store is not None and pending_meta.consent:
                store.add_turn(
                    pending_meta,
                    prompt_id=pending_turn["prompt_id"],
                    interrupt_label=pending_turn["interrupt_label"],
                    user_audio=pending_turn["user_audio"],
                    interaction_audio=session.controller.microphone_tail,
                    t_first_audio_ms=session.log.t_first_audio_ms,
                    server_generation_ms=session.log.server_generation_ms,
                    t_barge_in_ms=session.log.t_barge_in_ms,
                    stopped=session.log.stopped,
                )
            pending_turn = None

        def begin_interruption_continuation() -> None:
            """Carry captured barge-in pre-roll into the next user turn."""

            nonlocal buf, collecting
            if collecting:
                return
            pre_roll = session.controller.microphone_tail
            buf = [pre_roll] if pre_roll.size else []
            collecting = True

        def reset_transport(*, discard_pending: bool = False) -> None:
            """Return the socket to an idle, reusable state after cancellation/error."""

            nonlocal buf, collecting, pending_turn
            if session.controller.playing:
                session.controller.stop_playback("transport_reset")
            buf = []
            collecting = False
            if discard_pending:
                pending_turn = None
            session.log = TurnLog()

        try:
            while True:
                message = await ws.receive()
                if message.get("type") == "websocket.disconnect":
                    persist_pending()
                    return
                if "bytes" in message and message["bytes"] is not None:
                    raw = message["bytes"]
                    if not raw or len(raw) % 2:
                        await ws.send_json(
                            {"event": "error", "message": "invalid or empty PCM16 frame"}
                        )
                        continue
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if collecting:
                        buf.append(pcm)
                    if session.controller.playing:
                        session.on_mic_while_playing(pcm)
                        if not session.controller.playing:
                            begin_interruption_continuation()
                            await ws.send_json(
                                {
                                    "event": "stopped",
                                    "server_detection_ms": session.log.t_barge_in_ms,
                                }
                            )
                elif "text" in message and message["text"]:
                    try:
                        payload = json.loads(message["text"])
                    except json.JSONDecodeError:
                        await ws.send_json({"event": "error", "message": "invalid JSON event"})
                        continue
                    if not isinstance(payload, dict):
                        await ws.send_json(
                            {"event": "error", "message": "JSON event must be an object"}
                        )
                        continue
                    for key in (
                        "prompt_id",
                        "interrupt_label",
                        "session_id",
                        "speaker_id",
                        "age_bin",
                    ):
                        if payload.get(key):
                            turn_meta[key] = payload[key]
                    if "consent" in payload:
                        turn_meta["consent"] = payload["consent"] is True
                    local_meta = SessionMeta(
                        session_id=str(turn_meta["session_id"]),
                        speaker_id=str(turn_meta["speaker_id"]),
                        gpu_name=str(host_gpu.get("device") or "cpu"),
                        vram_gb=float(host_gpu["total_gb"])
                        if host_gpu.get("total_gb") is not None
                        else None,
                        memory_capped=bool(host_gpu.get("memory_capped", False)),
                        age_bin=str(turn_meta["age_bin"]),
                        consent=bool(turn_meta["consent"]),
                        retention=effective_retention,
                    )
                    event = payload.get("event")
                    if (
                        event in {"hello", "begin_utterance"}
                        and store is not None
                        and local_meta.consent
                    ):
                        try:
                            store.start(local_meta)
                        except (ValueError, PermissionError) as exc:
                            turn_meta["consent"] = False
                            local_meta.consent = False
                            await ws.send_json({"event": "error", "message": str(exc)})
                    if event == "begin_utterance":
                        if session.controller.playing:
                            session.controller.stop_playback("new_utterance")
                            session.log.stopped = True
                        persist_pending()
                        session.log = TurnLog()
                        buf = []
                        collecting = True
                    elif event == "interrupt":
                        session.controller.mark_interrupt_onset()
                        session.controller.stop_playback("client")
                        session.log.stopped = True
                        session.log.t_barge_in_ms = session.controller.t_barge_in_ms()
                        begin_interruption_continuation()
                        await ws.send_json(
                            {
                                "event": "stopped",
                                "server_detection_ms": session.log.t_barge_in_ms,
                            }
                        )
                    elif event == "cancel":
                        reset_transport(discard_pending=True)
                        await ws.send_json({"event": "cancelled"})
                    elif event == "end_of_speech":
                        collecting = False
                        if not buf:
                            await ws.send_json(
                                {"event": "error", "message": "no microphone audio received"}
                            )
                            continue
                        user = np.concatenate(buf)
                        try:
                            reply = session.on_user_end(user)
                        except Exception as exc:
                            error_name = (
                                "generation_out_of_memory"
                                if isinstance(exc, MemoryError)
                                or "out of memory" in str(exc).lower()
                                else "generation_failed"
                            )
                            reset_transport(discard_pending=True)
                            await ws.send_json({"event": "error", "message": error_name})
                            continue
                        pending_turn = {
                            "meta": local_meta,
                            "prompt_id": str(turn_meta.get("prompt_id") or "free"),
                            "interrupt_label": str(turn_meta.get("interrupt_label") or "none"),
                            "user_audio": user,
                        }
                        pcm16 = np.clip(reply * 32767, -32768, 32767).astype(np.int16)
                        await ws.send_bytes(pcm16.tobytes())
                        await ws.send_json(
                            {
                                "event": "reply_ready",
                                "server_generation_ms": session.log.server_generation_ms,
                                "talker": session.log.talker,
                                "tts_backend": session.log.tts_backend,
                                "reply_samples": int(len(reply)),
                                "transcript": getattr(talker, "last_transcript", None),
                                "reply_text": getattr(talker, "last_reply_text", None),
                                "asr_error": (
                                    "asr_request_failed"
                                    if getattr(talker, "last_asr_error", None)
                                    else None
                                ),
                                "responder_backend": getattr(talker, "responder_backend", None),
                                "responder_initialization_error": (
                                    "model_initialization_failed"
                                    if getattr(talker, "responder_initialization_error", None)
                                    else None
                                ),
                                "responder_fallback_used": getattr(
                                    talker, "last_responder_fallback_used", None
                                ),
                                "responder_error": (
                                    "model_generation_failed"
                                    if getattr(talker, "last_responder_error", None)
                                    else None
                                ),
                                "responder_generation_attempts": getattr(
                                    talker, "last_responder_generation_attempts", None
                                ),
                                "responder_language_retry_used": getattr(
                                    talker, "last_responder_language_retry_used", None
                                ),
                            }
                        )
                        buf = []
                    elif event == "playback_started":
                        session.log.t_first_audio_ms = valid_ms(payload.get("t_first_audio_ms"))
                    elif event == "playback_stopped_ack":
                        client_ms = valid_ms(payload.get("t_barge_in_ms"))
                        if client_ms is not None:
                            session.log.t_barge_in_ms = client_ms
                        session.log.stopped = True
                        persist_pending()
                        if collecting:
                            session.log = TurnLog()
                    elif event == "playback_ended":
                        session.controller.stop_playback("completed")
                        persist_pending()
                    elif event not in {"hello", None}:
                        await ws.send_json({"event": "error", "message": "unknown event"})
        except WebSocketDisconnect:
            persist_pending()
            return

    return app
