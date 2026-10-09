"""
수기/기계식 판별 웹 데모 (표준 라이브러리 http.server, 추가 설치 없음)

  python web_demo.py [--port 8000]   ->   http://localhost:8000

이미지를 올리면(드래그·붙여넣기 가능) 판정, 점수 S_hand, φ₁~φ₈, 유착 검출 박스를 바로 보여준다.
"""
import argparse
import base64
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import objective_function as of

MAX_UPLOAD = 25 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>수기 판별 데모</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#6b7280;--line:#e3e6ec;--hand:#d93025;--prt:#1e8e3e;--unc:#e37400;--bar:#3b6fd4}
@media(prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1e222a;--fg:#e8eaed;--mut:#9aa0a6;--line:#323843}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Malgun Gothic",sans-serif}
main{max-width:1000px;margin:0 auto;padding:20px 16px 48px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
#drop{border:2px dashed var(--line);border-radius:10px;padding:28px;text-align:center;color:var(--mut);cursor:pointer}
#drop.over{border-color:var(--bar);color:var(--bar)}
.verdict{display:inline-block;color:#fff;border-radius:8px;padding:6px 14px;font-weight:700;font-size:18px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:720px){.grid{grid-template-columns:1fr}}
img{max-width:100%;border-radius:6px;border:1px solid var(--line)}
.kv{display:flex;flex-wrap:wrap;gap:6px 22px;margin:10px 0;color:var(--mut)}.kv b{color:var(--fg)}
.row{display:grid;grid-template-columns:150px 1fr 56px;gap:8px;align-items:center;margin:5px 0}
.track{background:var(--line);border-radius:4px;height:12px}.fill{background:var(--bar);height:12px;border-radius:4px}
.flag{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:1px 10px;margin:2px 4px 2px 0;font-size:13px}
h2{font-size:15px;margin:0 0 8px}.muted{color:var(--mut)}
</style></head><body><main>
<h1>수기 / 기계식 마킹 판별</h1>
<p class="sub">철판 마킹 크롭 이미지를 올리면 1단계(기계 패턴) → 2단계(φ₁~φ₈ + 유착 가산점) 결과를 바로 보여줍니다.</p>
<div class="card"><div id="drop">이미지를 여기에 끌어놓거나, 클릭해서 선택하거나, 붙여넣기(Ctrl+V)</div>
<input id="file" type="file" accept="image/*" hidden></div>
<div id="out"></div>
<script>
const $=s=>document.querySelector(s),drop=$('#drop'),file=$('#file'),out=$('#out');
const COLOR={'Confirmed Handwritten':'var(--hand)','Confirmed Printed':'var(--prt)','Uncertain (Need Review)':'var(--unc)'};
const NAME={phi_1:'φ1 획 두께',phi_2:'φ2 간격',phi_3:'φ3 기울기',phi_4:'φ4 기준선',phi_5:'φ5 곡률',phi_6:'φ6 크기',phi_7:'φ7 외곽 요철',phi_8:'φ8 연결성'};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
drop.onclick=()=>file.click();file.onchange=()=>file.files[0]&&run(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');e.dataTransfer.files[0]&&run(e.dataTransfer.files[0])};
addEventListener('paste',e=>{const f=[...e.clipboardData.files][0];f&&run(f)});
async function run(f){
  out.innerHTML='<div class="card muted">분석 중…</div>';
  try{
    const r=await fetch('/analyze',{method:'POST',body:f});const d=await r.json();
    if(!r.ok)throw new Error(d.error||r.status);show(d)
  }catch(e){out.innerHTML='<div class="card">오류: '+esc(e.message)+'</div>'}
}
function show(d){
  const s=d.score==null?'n/a':d.score.toFixed(3);
  let h='<div class="card"><span class="verdict" style="background:'+COLOR[d.verdict]+'">'+esc(d.verdict)+'</span>'
   +'<div class="kv"><span>단계 <b>'+d.stage+'</b></span><span>S_hand <b>'+s+'</b></span><span>참고 신뢰도 <b>'+d.confidence.toFixed(2)+'</b></span>'
   +'<span>임계값 인쇄 ≤ <b>'+d.thresholds.printed+'</b> / 수기 ≥ <b>'+d.thresholds.handwritten+'</b></span></div>';
  h+=d.flags.map(f=>'<span class="flag">'+esc(f)+'</span>').join('');
  if(d.stage==1)h+='<p class="muted">기계 패턴('+esc(d.mechanical_kind)+')이 검출되어 φ 계산 없이 인쇄체로 확정했습니다.</p>';
  if(d.stage==0)h+='<p class="muted">반사광 포화로 점수를 계산하지 않았습니다.</p>';
  if(d.stage==2){
    h+='<p class="muted">유착 검출: <b>'+(d.has_touching_chars?'있음':'없음')+'</b>'
     +(d.has_touching_chars?' (가산점 b='+d.touch_boost+', 가산 전 S='+d.score_before_touch.toFixed(3)+')':'')+'</p>';
  }
  h+='</div><div class="grid"><div class="card"><h2>유착 검출 결과 (빨간 박스)</h2><img src="data:image/png;base64,'+d.annotated+'"></div>'
   +'<div class="card"><h2>φ 지표 (0=인쇄체 쪽, 1=수기 쪽)</h2>';
  for(const k in NAME){const v=d.features[k],w=d.weights_used[k];
    h+='<div class="row"><span>'+NAME[k]+'</span><div class="track">'+(v==null?'':'<div class="fill" style="width:'+(v*100)+'%"></div>')
      +'</div><span class="muted">'+(v==null?'n/a':v.toFixed(2))+'</span></div>'}
  h+='<p class="muted">막대 옆 숫자는 φ 값, 가중치는 objective_weights.json 기준</p></div></div>'
   +'<div class="card"><h2>상세 시각화 (레이더·가중치)</h2><img src="data:image/png;base64,'+d.debug+'"></div>';
  out.innerHTML=h
}
</script></main></body></html>"""


def _png_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode()


def analyze(data: bytes) -> dict:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없습니다")
    r = of.evaluate_handwritten_score(img)
    canvas = img.copy()
    t = r["touching"]
    scale = 1 / r["work_scale"]                       # 작업 해상도 -> 원본 좌표
    for c in (t["components"] if t else []):
        x, y, w, h = (int(v * scale) for v in c["box"])
        cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 230), max(2, canvas.shape[1] // 300))
    return {
        "verdict": r["verdict"], "stage": r["stage"], "score": r["score"], "confidence": r["confidence"],
        "thresholds": r["thresholds"], "flags": r["flags"],
        "mechanical_kind": r["mechanical"]["kind"] if r["mechanical"] else None,
        "has_touching_chars": r["has_touching_chars"], "touch_boost": r.get("touch_boost", 0.0),
        "score_before_touch": r.get("score_before_touch"),
        "features": r["features"], "weights_used": r["weights_used"],
        "annotated": _png_b64(canvas), "debug": _png_b64(of.draw_objective_debug(img, r)),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/analyze":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_UPLOAD:
            return self._send(413, json.dumps({"error": "파일이 없거나 너무 큽니다 (25MB 이하)"}).encode(), "application/json")
        try:
            body, code = json.dumps(analyze(self.rfile.read(n))).encode(), 200
        except Exception as e:                       # 사용자에게 사유를 보여준다
            body, code = json.dumps({"error": str(e)}).encode("utf-8"), 400
        self._send(code, body, "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="수기 판별 웹 데모")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)     # 로컬 전용
    url = f"http://localhost:{args.port}"
    print(f"웹 데모: {url}  (종료: Ctrl+C)")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()
