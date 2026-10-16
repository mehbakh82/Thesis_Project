#!/usr/bin/env python3
"""Build the minimal remote listening study as one self-contained HTML file.

Participants open the file on a phone or computer (no server, no recording):
consent and age group, then naturalness ratings (1-5 absolute category rating)
of short clips in a per-participant random order, an attention check, and three
recorded interactions with the live system rated for responsiveness,
interruption handling, satisfaction, and willingness to use. The page ends with
a short result code to send back; `scripts/ingest_listening_study.py` decodes it.
No names or contact data are collected.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ارزیابی شنیداری دستیار صوتی فارسی</title>
<style>
:root{--bg:#fafafa;--fg:#1b1b1b;--card:#fff;--accent:#2e6b3f;--muted:#666}
@media (prefers-color-scheme: dark){:root{--bg:#141414;--fg:#eee;--card:#1f1f1f;--accent:#79c08f;--muted:#aaa}}
body{font-family:Tahoma,Vazirmatn,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;font-size:1.15rem;line-height:1.9}
.card{background:var(--card);border-radius:14px;padding:16px;margin:12px 0;box-shadow:0 1px 4px #0002}
button{font-size:1.15rem;padding:12px 18px;border-radius:12px;border:0;background:var(--accent);color:#fff;margin:6px 4px}
.scale button{background:#ddd;color:#111;min-width:52px}.scale button.on{background:var(--accent);color:#fff}
audio{width:100%;margin:8px 0}.muted{color:var(--muted);font-size:1rem}
textarea{width:100%;font-size:1.05rem;min-height:80px}
#code{direction:ltr;word-break:break-all;font-family:monospace;font-size:.95rem;background:#0001;padding:10px;border-radius:8px}
</style></head><body>
<div id="app"></div>
<script>
const STUDY = __STUDY__;
const LABELS = ["بد","ضعیف","متوسط","خوب","عالی"];
const app = document.getElementById('app');
const answers = {study: STUDY.id, ratings: {}, interactions: {}};
let order = [];
function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function scale(name, onPick){
  const d=document.createElement('div'); d.className='scale';
  for(let v=1; v<=5; v++){const b=document.createElement('button'); b.textContent=v+' — '+LABELS[v-1];
    b.onclick=()=>{[...d.children].forEach(c=>c.classList.remove('on')); b.classList.add('on'); onPick(v);}; d.appendChild(b);}
  return d;
}
function screenIntro(){
  app.innerHTML = `<div class="card"><h2>ارزیابی شنیداری یک دستیار صوتی فارسی</h2>
  <p>این آزمون حدود ۱۵ دقیقه طول می‌کشد. چند صدای کوتاه می‌شنوید و به طبیعی بودن آن‌ها نمره می‌دهید، سپس سه گفت‌وگوی ضبط‌شده با دستیار را گوش می‌کنید.</p>
  <p>هیچ صدایی از شما ضبط نمی‌شود و نام یا اطلاعات تماس شما پرسیده نمی‌شود. در هر زمان می‌توانید آزمون را رها کنید.</p>
  <p class="muted">لطفاً اگر ممکن است از هدفون استفاده کنید و صدای گوشی را روی حد راحت تنظیم کنید.</p>
  <label><input type="checkbox" id="consent"> توضیحات بالا را خواندم و با شرکت در این آزمون موافقم.</label>
  <p>گروه سنی:
   <select id="age"><option value="">انتخاب کنید</option><option value="under_40">زیر ۴۰</option><option value="40_59">۴۰ تا ۵۹</option><option value="60plus">۶۰ و بالاتر</option></select></p>
  <p>زبان اول شما فارسی است؟ <select id="native"><option value="yes">بله</option><option value="no">خیر</option></select></p>
  <p>شنیدن با: <select id="device"><option value="headphones">هدفون</option><option value="speaker">بلندگوی گوشی/رایانه</option></select></p>
  <button id="go">شروع</button></div>`;
  document.getElementById('go').onclick=()=>{
    if(!document.getElementById('consent').checked || !document.getElementById('age').value){alert('لطفاً رضایت و گروه سنی را مشخص کنید.');return;}
    answers.consent=true; answers.age=document.getElementById('age').value; answers.native=document.getElementById('native').value; answers.device=document.getElementById('device').value;
    order = shuffle(STUDY.clips.map(c=>c.id)); screenClip(0);
  };
}
function screenClip(i){
  if(i>=order.length){screenInteraction(0);return;}
  const clip=STUDY.clips.find(c=>c.id===order[i]);
  app.innerHTML=`<div class="card"><p class="muted">صدای ${i+1} از ${order.length}</p>
  <p>${clip.attention ? 'در این صدا از شما خواسته می‌شود گزینه‌ای را انتخاب کنید؛ همان را انتخاب کنید.' : 'این صدا چقدر <b>طبیعی</b> و شبیه گفتار انسان است؟'}</p>
  <audio controls src="${clip.src}"></audio><div id="s"></div>
  <button id="next" disabled>بعدی</button></div>`;
  document.getElementById('s').appendChild(scale(clip.id, v=>{answers.ratings[clip.id]=v; document.getElementById('next').disabled=false;}));
  document.getElementById('next').onclick=()=>screenClip(i+1);
}
function screenInteraction(i){
  if(i>=STUDY.interactions.length){screenEnd();return;}
  const it=STUDY.interactions[i]; const res={};
  app.innerHTML=`<div class="card"><p class="muted">گفت‌وگوی ${i+1} از ${STUDY.interactions.length}</p>
  <p>${it.description}</p><audio controls src="${it.src}"></audio>
  <p>سرعت پاسخ‌گویی دستیار چطور بود؟</p><div id="q1"></div>
  ${it.has_interruption?'<p>وقتی کاربر وسط حرف دستیار صحبت کرد، دستیار چقدر خوب واکنش نشان داد؟</p><div id="q2"></div>':''}
  <p>در مجموع چقدر از این گفت‌وگو راضی هستید؟</p><div id="q3"></div>
  <p>دوست دارید از چنین دستیاری استفاده کنید؟ <select id="use"><option value="">انتخاب کنید</option><option value="yes">بله</option><option value="no">خیر</option></select></p>
  <button id="next">بعدی</button></div>`;
  document.getElementById('q1').appendChild(scale('r',v=>res.responsiveness=v));
  if(it.has_interruption) document.getElementById('q2').appendChild(scale('i',v=>res.interruption=v));
  document.getElementById('q3').appendChild(scale('s',v=>res.satisfaction=v));
  document.getElementById('next').onclick=()=>{
    res.would_use=document.getElementById('use').value;
    if(!res.responsiveness||!res.satisfaction||!res.would_use||(it.has_interruption&&!res.interruption)){alert('لطفاً به همهٔ پرسش‌ها پاسخ دهید.');return;}
    answers.interactions[it.id]=res; screenInteraction(i+1);};
}
function screenEnd(){
  app.innerHTML=`<div class="card"><p>اگر نکته‌ای دربارهٔ صدای دستیار یا استفادهٔ افراد مسن از آن دارید بنویسید (اختیاری):</p>
  <textarea id="notes" maxlength="800"></textarea><button id="fin">پایان و دریافت کد</button></div>`;
  document.getElementById('fin').onclick=()=>{
    answers.notes=document.getElementById('notes').value.slice(0,800);
    const code=btoa(unescape(encodeURIComponent(JSON.stringify(answers))));
    app.innerHTML=`<div class="card"><h3>سپاسگزاریم!</h3><p>لطفاً کد زیر را کامل کپی کنید و برای پژوهشگر بفرستید:</p>
    <div id="code">${code}</div><button id="copy">کپی کد</button></div>`;
    document.getElementById('copy').onclick=()=>navigator.clipboard.writeText(code).then(()=>alert('کپی شد'));
  };
}
screenIntro();
</script></body></html>
"""


def mp3_data_uri(path: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "clip.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-ac", "1", "-ar", "22050", "-b:a", "48k", str(out)],
            check=True,
        )
        return "data:audio/mpeg;base64," + base64.b64encode(out.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="JSON: clips[] and interactions[] with local paths")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    clips = [
        {"id": c["id"], "src": mp3_data_uri(Path(c["path"])), "attention": bool(c.get("attention"))}
        for c in spec["clips"]
    ]
    interactions = [
        {
            "id": it["id"],
            "src": mp3_data_uri(Path(it["path"])),
            "description": it["description"],
            "has_interruption": bool(it.get("has_interruption")),
        }
        for it in spec["interactions"]
    ]
    study = {"id": spec["study_id"], "clips": clips, "interactions": interactions}
    html = PAGE.replace("__STUDY__", json.dumps(study, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(json.dumps({"clips": len(clips), "interactions": len(interactions), "bytes": len(html.encode()), "sha256": hashlib.sha256(html.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
