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
                    raw = open(f).read()
                    head, bs, fs = (raw.split("|") + ["", ""])[:3]
                    parts = head.split(",")
                    x, y, stype, chain = parts[:4]
                    hammer = int(parts[4]) if len(parts) > 4 else 0
                    pts = lambda s: [[int(a) for a in pair.split(":")]
                                     for pair in s.split(";") if ":" in pair]
                    out.append({"port": p, "x": int(x), "y": int(y),
                                "t": stype, "c": int(chain),
                                "b": pts(bs), "fb": pts(fs), "h": hammer,
                                # envs pause several seconds during PPO updates
                                "stale": now - st.st_mtime > 15})
                except (OSError, ValueError):
                    out.append({"port": p, "stale": True})
            ages = [now - os.stat(f"/dev/shm/dk_live_{p}").st_mtime
                    for p in PORTS if os.path.exists(f"/dev/shm/dk_live_{p}")]
            # all envs silent >0.8s but not dead => PPO update in progress
            learning = bool(ages) and min(ages) > 0.8 and min(ages) < 15
            body = json.dumps({"envs": out, "learning": learning}).encode()
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
body { background:#0D0B14; color:#E2DEEE; font:13px system-ui; margin:0; padding:14px; }
#grid { display:grid; grid-template-columns:repeat(4, max-content); gap:14px; }
.cell { position:relative; }
.cell img.bg { display:block; width:320px; image-rendering:pixelated; }
.cell svg { position:absolute; inset:0; width:100%; height:100%; }
.cell .tag { position:absolute; top:4px; left:6px; font:600 11px ui-monospace,monospace;
  color:#F2B33D; letter-spacing:.12em; text-shadow:0 0 4px #000; }
#bar { display:flex; gap:22px; margin-bottom:10px; font-family:ui-monospace,monospace;
  font-size:12px; align-items:baseline; }
#bar h1 { font:600 14px ui-monospace,monospace; letter-spacing:.15em;
  color:#F2B33D; margin:0 14px 0 0; }
.sw { display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:5px; vertical-align:-1px; }
</style>
<div id=bar>
  <h1>DK LIVE &middot; 16 ENVS &middot; 8 BOARDS</h1>
  <span><span class=sw style="background:#F2B33D"></span>bottom-up</span>
  <span><span class=sw style="background:#7BD88F"></span>floor/ladder chains</span>
  <span><span class=sw style="background:#6FC3D6"></span>tower chains</span>
  <span id=stat style="color:#8B85A3"></span>
  <span id=learn style="display:none;color:#0D0B14;background:#F2B33D;
    padding:2px 10px;border-radius:3px;font-weight:600;letter-spacing:.12em">LEARNING&hellip;</span>
</div>
<div id=grid></div>
<script>
const NS='http://www.w3.org/2000/svg';
const MARIO='data:image/png;base64,__MARIO__';
const BARREL='data:image/png;base64,__BARREL__';
const FIRE='data:image/png;base64,__FIRE__';
const HAMMER='data:image/png;base64,__HAMMER__';
const BG='data:image/jpeg;base64,__BG__';
const ix=x=>(x-14.5)*3, iy=y=>(y-7.5)*3;
const grid=document.getElementById('grid'), panels=[];
const GROUPS=[[5000,5001],[5002,5003],[5004,5005],[5006,5007],
              [5008,5009],[5010,5011],[5012,5013],[5014,5015]];
const P2PANEL={}; GROUPS.forEach((g,i)=>g.forEach(p=>P2PANEL[p]=i));
for(let i=0;i<8;i++){
  const g=GROUPS[i];
  const cell=document.createElement('div'); cell.className='cell';
  cell.innerHTML='<img class=bg src="'+BG+'">'
    +'<svg viewBox="0 0 672 768" preserveAspectRatio="none"></svg>'
    +'<div class=tag>ENVS '+g[0]+'-'+g[g.length-1]+'</div>';
  grid.appendChild(cell);
  panels.push(cell.querySelector('svg'));
}
const E={};   // per-port entity state with lerp targets
function el(svg,t,a){const e=document.createElementNS(NS,t);
  for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;}
async function poll(){
  try{
    const resp=await (await fetch('/state')).json();
    const st=resp.envs||resp;
    const lb=document.getElementById('learn');
    lb.style.display=resp.learning?'inline-block':'none';
    document.getElementById('grid').style.opacity=resp.learning?.55:1;
    let live=0;
    for(const s of st){
      const sv=panels[P2PANEL[s.port]||0];
      if(!E[s.port]){
        E[s.port]={tr:el(sv,'polyline',{fill:'none','stroke-width':1.5,opacity:.45}),
          ring:el(sv,'ellipse',{rx:18,ry:5,fill:'none','stroke-width':2.5}),
          m:el(sv,'image',{href:MARIO,width:40,height:56}),
          th:[],sv:sv,cur:null,tgt:null,thT:[]};
      }
      const e=E[s.port];
      if(s.stale){e.m.setAttribute('opacity',.15);e.ring.setAttribute('opacity',.15);
        e.tr.setAttribute('points','');e.th.forEach(t=>t.setAttribute('opacity',0));
        e.cur=e.tgt=null;continue;}
      live++;
      const px=ix(s.x), py=iy(s.y);
      e.tgt={x:px,y:py,h:s.h,col:s.t==='bottomup'?'#F2B33D':(s.c>=12?'#7BD88F':'#6FC3D6')};
      if(!e.cur||Math.hypot(px-e.cur.x,py-e.cur.y)>120){
        e.cur={x:px,y:py};e.tr.setAttribute('points','');}
      const objs=(s.b||[]).map(p=>({p,href:BARREL,w:18,h:20,dy:24}))
        .concat((s.fb||[]).map(p=>({p,href:FIRE,w:20,h:23,dy:30})));
      while(e.th.length<objs.length)e.th.push(el(sv,'image',{opacity:0}));
      while(e.thT.length<objs.length)e.thT.push(null);
      e.th.forEach((t,i)=>{
        if(i>=objs.length){t.setAttribute('opacity',0);e.thT[i]=null;return;}
        const o=objs[i], tx=ix(o.p[0]), ty=iy(o.p[1]);
        t.setAttribute('href',o.href);t.setAttribute('width',o.w);t.setAttribute('height',o.h);
        const prev=e.thT[i];
        e.thT[i]={x:tx,y:ty,w:o.w,h:o.h,dy:o.dy,
                  cx:prev&&Math.hypot(tx-prev.cx,ty-prev.cy)<80?prev.cx:tx,
                  cy:prev&&Math.hypot(tx-prev.cx,ty-prev.cy)<80?prev.cy:ty};
        t.setAttribute('opacity',.9);
      });
      const pts=(e.tr.getAttribute('points')||'').split(' ').filter(Boolean);
      pts.push(px+','+py); if(pts.length>60)pts.shift();
      e.tr.setAttribute('points',pts.join(' '));
      e.tr.setAttribute('stroke',e.tgt.col);
    }
    document.getElementById('stat').textContent=live+'/16 live';
  }catch(err){document.getElementById('stat').textContent='server unreachable';}
  setTimeout(poll,80);
}
function render(){
  const L=0.28;
  for(const p in E){
    const e=E[p];
    if(e.tgt&&e.cur){
      e.cur.x+=(e.tgt.x-e.cur.x)*L; e.cur.y+=(e.tgt.y-e.cur.y)*L;
      const px=e.cur.x, py=e.cur.y;
      if(e.tgt.h){e.m.setAttribute('href',HAMMER);e.m.setAttribute('width',66);
        e.m.setAttribute('height',38);e.m.setAttribute('x',px-44);e.m.setAttribute('y',py+4);}
      else{e.m.setAttribute('href',MARIO);e.m.setAttribute('width',40);
        e.m.setAttribute('height',56);e.m.setAttribute('x',px-20);e.m.setAttribute('y',py-14);}
      e.m.setAttribute('opacity',1);
      e.ring.setAttribute('cx',px);e.ring.setAttribute('cy',py+40);
      e.ring.setAttribute('stroke',e.tgt.col);e.ring.setAttribute('opacity',.9);
      e.th.forEach((t,i)=>{
        const o=e.thT[i]; if(!o)return;
        o.cx+=(o.x-o.cx)*L; o.cy+=(o.y-o.cy)*L;
        t.setAttribute('x',o.cx-o.w/2);t.setAttribute('y',o.cy-o.h/2+o.dy);
      });
    }
  }
  requestAnimationFrame(render);
}
poll(); requestAnimationFrame(render);
</script>"""
_art = os.path.join(os.path.dirname(__file__), "..", "artifacts")
HTML = HTML.replace("__BG__", open(os.path.join(_art, "live_bg_b64.txt")).read().strip())
HTML = HTML.replace("__MARIO__", open(os.path.join(_art, "live_mario_b64.txt")).read().strip())
HTML = HTML.replace("__BARREL__", open(os.path.join(_art, "live_barrel_b64.txt")).read().strip())
HTML = HTML.replace("__FIRE__", open(os.path.join(_art, "live_fire_b64.txt")).read().strip())
HTML = HTML.replace("__HAMMER__", open(os.path.join(_art, "live_hammer_b64.txt")).read().strip())

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8600
    print(f"live board on http://localhost:{port}")
    http.server.ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
