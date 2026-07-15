#!/usr/bin/env python3
"""Live spectator board: 16 Mario dots on the real annotated board.
Reads /dev/shm/dk_live_<port> (written by mame_env.step) at ~7Hz.
Run:  python3 scripts/live_board.py [port]   then open http://localhost:8600
"""
import http.server, json, os, sys, time

PORTS = list(range(5000, 5016))
HTML = None  # loaded below

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/state":
            out = []
            now = time.time()
            for p in PORTS:
                f = f"/dev/shm/dk_live_{p}"
                try:
                    st = os.stat(f)
                    x, y, stype, chain = open(f).read().split(",")
                    out.append({"port": p, "x": int(x), "y": int(y),
                                "t": stype, "c": int(chain),
                                "stale": now - st.st_mtime > 15})  # envs pause
                                # several seconds during each PPO update
                except (OSError, ValueError):
                    out.append({"port": p, "stale": True})
            body = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

HTML = r"""<!doctype html><title>DK Live Board</title>
<style>
body { background:#0D0B14; color:#E2DEEE; font:13px system-ui; margin:0;
  display:flex; gap:16px; padding:16px; }
#box { position:relative; }
#box img.bg { display:block; width:560px; image-rendering:pixelated; }
#box svg { position:absolute; inset:0; width:100%; height:100%; }
aside { font-family:ui-monospace,monospace; font-size:12px; line-height:1.9; }
h1 { font:600 14px ui-monospace,monospace; letter-spacing:.15em; color:#F2B33D; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%;
  margin-right:6px; vertical-align:-1px; }
</style>
<div id=box>
  <img class=bg src="data:image/jpeg;base64,__BG__">
  <svg id=ov viewBox="0 0 672 768" preserveAspectRatio="none"></svg>
</div>
<aside>
  <h1>DK LIVE &middot; 16 ENVS</h1>
  <div><span class=sw style="background:#F2B33D"></span>bottom-up (cold start)</div>
  <div><span class=sw style="background:#7BD88F"></span>floor / ladder chains 12-15</div>
  <div><span class=sw style="background:#6FC3D6"></span>tower chains 0-11</div>
  <div><span class=sw style="background:#666"></span>stale / resetting</div>
  <div id=stat style="margin-top:12px;color:#8B85A3"></div>
</aside>
<script>
const NS='http://www.w3.org/2000/svg', ov=document.getElementById('ov');
const MARIO='data:image/png;base64,__MARIO__';
// native frame: screen = RAM - (14.5, 7.5); display scale x3
const ix=x=>(x-14.5)*3, iy=y=>(y-7.5)*3;
const marks={}, rings={}, trails={};
function el(t,a){const e=document.createElementNS(NS,t);
  for(const k in a)e.setAttribute(k,a[k]);ov.appendChild(e);return e;}
async function tick(){
  try{
    const st=await (await fetch('/state')).json();
    let live=0;
    for(const s of st){
      if(!marks[s.port]){
        trails[s.port]=el('polyline',{fill:'none','stroke-width':1.5,opacity:.45});
        rings[s.port]=el('ellipse',{rx:16,ry:5,fill:'none','stroke-width':2.5});
        marks[s.port]=el('image',{href:MARIO,width:32,height:48});
      }
      const m=marks[s.port], ring=rings[s.port], tr=trails[s.port];
      if(s.stale){m.setAttribute('opacity',.15);ring.setAttribute('opacity',.15);
        tr.setAttribute('points','');continue;}
      live++;
      const col=s.t==='bottomup'?'#F2B33D':(s.c>=12?'#7BD88F':'#6FC3D6');
      const px=ix(s.x), py=iy(s.y);
      m.setAttribute('x',px-16);m.setAttribute('y',py-24);m.setAttribute('opacity',1);
      ring.setAttribute('cx',px);ring.setAttribute('cy',py+24);
      ring.setAttribute('stroke',col);ring.setAttribute('opacity',.9);
      tr.setAttribute('stroke',col);
      const pts=(tr.getAttribute('points')||'').split(' ').filter(Boolean);
      pts.push(px+','+py); if(pts.length>40)pts.shift();
      if(pts.length>1){const [lx,ly]=pts[pts.length-2].split(',').map(Number);
        if(Math.hypot(px-lx,py-ly)>120)pts.splice(0,pts.length-1);}
      tr.setAttribute('points',pts.join(' '));
    }
    document.getElementById('stat').textContent=live+'/16 live';
  }catch(e){document.getElementById('stat').textContent='server unreachable';}
  setTimeout(tick,140);
}
tick();
</script>"""
_art = os.path.join(os.path.dirname(__file__), "..", "artifacts")
HTML = HTML.replace("__BG__", open(os.path.join(_art, "live_bg_b64.txt")).read().strip())
HTML = HTML.replace("__MARIO__", open(os.path.join(_art, "live_mario_b64.txt")).read().strip())

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
    print(f"live board on http://localhost:{port}")
    http.server.ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
