#!/usr/bin/env python3
"""Live spectator board: 16 Mario dots on the real annotated board.
Reads /dev/shm/dk_live_<port> (written by mame_env.step) at ~7Hz.
Run:  python3 scripts/live_board.py [port]   then open http://localhost:8600
"""
import csv, glob, http.server, json, os, sys, time

PORTS = list(range(5000, 5016))
_MCACHE = {"t": 0, "data": {}}

def metrics():
    if time.time() - _MCACHE["t"] < 8:
        return _MCACHE["data"]
    now = time.time()
    rows = []
    for f in glob.glob("logs/episodes/dk_*.monitor.csv"):
        try:
            with open(f) as fh:
                fh.readline()
                for row in csv.DictReader(fh):
                    rows.append(row)
        except OSError:
            pass
    def F(r, k, d=0.0):
        try: return float(r.get(k) or d)
        except (TypeError, ValueError): return d
    bu = [r for r in rows if r.get("start_type") == "bottomup"]
    bu_clean = [r for r in bu if F(r,"glitch_kill")==0 and F(r,"no_barrels")==0]
    clears = sum(1 for r in bu_clean if F(r,"cleared") == 1)
    ws = [r for r in rows if r.get("bw_chain") in ("14","15")
          and F(r,"glitch_kill")==0 and F(r,"no_barrels")==0]
    wsg = [F(r,"max_height")-(240-F(r,"start_y")) for r in ws]
    fl = [r for r in rows if r.get("bw_chain") in ("12","13")
          and F(r,"glitch_kill")==0 and F(r,"no_barrels")==0]
    flg = [F(r,"max_height")-(240-F(r,"start_y")) for r in fl]
    low = [r for r in rows if (r.get("start_type")=="bottomup" or
           r.get("bw_chain") in ("12","13","14","15"))
           and F(r,"glitch_kill")==0 and F(r,"no_barrels")==0
           and (240-F(r,"start_y")) < 50]
    lh = [F(r,"max_height") for r in low]
    try:
        with open("artifacts/backward_dense14/levels.json") as _lf:
            gates = sum(json.load(_lf)["levels"])
    except OSError:
        gates = -1
    try:
        with open("logs/run_label") as _rl:
            label = _rl.read().strip()
    except OSError:
        label = "?"
    d = {
        "label": label,
        "episodes": len(rows),
        "clears": clears,
        "bu_n": len(bu_clean),
        "bu_mean": round(sum(F(r,"max_height") for r in bu_clean)/max(len(bu_clean),1),1),
        "bu_max": int(max((F(r,"max_height") for r in bu_clean), default=0)),
        "ws_rate": round(100*sum(g>=20 for g in wsg)/max(len(wsg),1), 2),
        "ws_best": int(max(wsg, default=0)),
        "fl_rate": round(100*sum(g>=40 for g in flg)/max(len(flg),1), 2),
        "fl_best": int(max(flg, default=0)),
        "h63": sum(h>=63 for h in lh), "h65": sum(h>=65 for h in lh),
        "h68": sum(h>=68 for h in lh),
        "top": int(max(lh, default=0)),
        "gates": gates,
    }
    from collections import Counter
    # ladder forensics: deaths within a ladder column, split by phase
    LADS = [(203, 4, 29), (53, 44, 62), (131, 82, 122),
            (94, 118, 148), (67, 115, 155), (147, 140, 192)]
    lad = {}
    for lx, base, top in LADS:
        near = [e for e in TRK.deaths
                if abs(e["x"] - lx) <= 10 and base - 6 <= e["h"] <= top + 6]
        lad[f"x{lx}"] = {
            "mount": sum(1 for e in near if e["h"] < base + 7),
            "mid": sum(1 for e in near if base + 7 <= e["h"] <= top - 7),
            "top": sum(1 for e in near if e["h"] > top - 7)}
    d["ladders"] = lad
    legs = Counter(e["leg"] for e in TRK.deaths)
    d["legs"] = [[name, legs.get(name, 0)] for name, _ in LEGS]
    d.update({"hammer_pickups": TRK.stats["pickups"],
              "expiry_deaths": TRK.stats["expiry_deaths"],
              "d_barrel": TRK.stats["deaths_barrel"],
              "d_fireball": TRK.stats["deaths_fireball"],
              "d_self": TRK.stats["deaths_self"],
              "guard_kills": TRK.stats["guard_kills"],
              "commits": TRK.stats["commits"],
              "commits_clear_pct": round(100*TRK.stats["commits_clear"]/max(TRK.stats["commits"],1)),
              "commit_survive": TRK.stats["commit_survive"],
              "commit_split": {k: TRK.stats.get(k, 0) for k in
                               ("c_ws", "c_ws_clear", "c_bu", "c_bu_clear")}})
    _MCACHE["t"] = now; _MCACHE["data"] = d
    return d
HTML = None  # loaded below

# ---- streaming analytics: episode ends, legs, hammer economics ----
LEGS = [("floor walk", lambda x,h: h < 26 and x < 180),
        ("x203 approach/climb", lambda x,h: h < 33),
        ("g2 walk", lambda x,h: 33 <= h < 44 and x > 70),
        ("wait + x53 climb", lambda x,h: 33 <= h < 62 and x <= 70),
        ("girder 3 (kill zone)", lambda x,h: 44 <= h < 82),
        ("x131 climb / g4", lambda x,h: 82 <= h < 135),
        ("g5 / upper", lambda x,h: 135 <= h < 168),
        ("top section", lambda x,h: h >= 168)]
def leg_of(x, h):
    for name, fn in LEGS:
        if fn(x, h): return name
    return "other"

class Tracker:
    def __init__(self):
        self.last = {}          # port -> (x, y, hammer, glitch, t, score)
        self.h_expiry = {}      # port -> t of last hammer 1->0
        self.deaths = []        # ring buffer of end events
        self._label = ""        # current run label, TTL-cached
        self._label_t = 0.0
        self.stats = {"pickups": 0, "expiry_deaths": 0, "guard_kills": 0,
                      "deaths_barrel": 0, "deaths_fireball": 0, "deaths_self": 0,
                      "commits": 0, "commits_clear": 0, "commit_survive": 0}
        self.climb = {}         # port -> (mount_t, gap_was_clear)
        try:
            with open("/dev/shm/dk_analytics.json") as _af:
                d = json.load(_af)
            self.deaths = d.get("deaths", [])
            # Session counters (pickups, expiry deaths, ...) belong to ONE
            # run letter (user request 2026-07-16): only restore them if the
            # ledger was written by the run we're still in.
            if d.get("run") == self.label():
                self.stats.update(d.get("stats", {}))
        except (OSError, ValueError):
            pass
        self._saved = time.time()

    def label(self):
        now = time.time()
        if now - self._label_t > 30:
            self._label_t = now
            try:
                with open("logs/run_label") as _rl:
                    lab = _rl.read().strip()
            except OSError:
                return self._label
            if lab != self._label:
                if self._label:  # run letter changed -> new session counters
                    self.stats = {k: 0 for k in self.stats}
                    self.climb.clear()
                self._label = lab
        return self._label

    def feed(self, port, x, y, hammer, glitch, score, gap, now, stype="?",
             cause="?"):
        h = 240 - y
        prev = self.last.get(port)
        if prev:
            px, py, ph, pg, pt, ps = prev
            if hammer and not ph: self.stats["pickups"] += 1
            if ph and not hammer: self.h_expiry[port] = now
            # mount detection: entering the x53 column ascending
            # episode end FIRST: a reset teleport must clear climb state
            # before the survive check (a death->tower reload otherwise
            # counts as 'survived to g3'; caught 2026-07-15).
            if abs(x - px) + abs(y - py) > 45:
                self.climb.pop(port, None)
            elif (46 <= x <= 60 and py - y >= 2 and 178 <= y <= 198
                    and port not in self.climb):
                self.climb[port] = (now, gap > 0)
                self.stats["commits"] += 1
                if gap > 0: self.stats["commits_clear"] += 1
                k = "c_" + ("ws" if stype == "curriculum" else "bu")
                self.stats[k] = self.stats.get(k, 0) + 1
                if gap > 0:
                    self.stats[k + "_clear"] = self.stats.get(k + "_clear", 0) + 1
            elif port in self.climb and h >= 64:
                self.stats["commit_survive"] += 1
                del self.climb[port]
            # episode end: position teleport
            if abs(x - px) + abs(y - py) > 45:
                # cause is STICKY env truth written at the done step, so
                # the CURRENT tap (first post-teleport frame) carries the
                # cause of the end we are detecting right now.
                ev = {"x": px, "h": 240 - py, "t": round(now, 1),
                      "run": self.label(), "cause": cause,
                      "leg": leg_of(px, 240 - py), "glitch": pg,
                      "hx": bool(self.h_expiry.get(port) and
                                 now - self.h_expiry[port] < 2.5)}
                if pg: self.stats["guard_kills"] += 1
                _ck = {"b": "deaths_barrel", "f": "deaths_fireball",
                       "s": "deaths_self"}.get(cause)
                if _ck and not pg: self.stats[_ck] += 1
                if ev["hx"]: self.stats["expiry_deaths"] += 1
                self.deaths.append(ev)
                if len(self.deaths) > 6000: self.deaths = self.deaths[-5000:]
                self.climb.pop(port, None)
        self.last[port] = (x, y, hammer, glitch, now, score)
        if now - self._saved > 60:
            self._saved = now
            try:
                json.dump({"deaths": self.deaths[-5000:], "stats": self.stats,
                           "run": self._label},
                          open("/dev/shm/dk_analytics.json", "w"))
            except OSError:
                pass

TRK = Tracker()

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/deaths":
            # Death map shows the CURRENT run letter only (user request
            # 2026-07-16: "what is happening now as opposed to days ago").
            # The full ring stays in the ledger for forensics; pre-stamp
            # entries have no "run" key and age out of the view naturally.
            lab = TRK.label()
            body = json.dumps([e for e in TRK.deaths
                               if e.get("run") == lab][-2500:]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = json.dumps(metrics()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
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
                    score = int(parts[5]) if len(parts) > 5 else 0
                    gap = float(parts[6]) if len(parts) > 6 else 0
                    glitch = int(parts[7]) if len(parts) > 7 else 0
                    cause = parts[8] if len(parts) > 8 else "?"
                    if now - st.st_mtime < 15:
                        TRK.feed(p, int(x), int(y), hammer, glitch, score,
                                 gap, now, stype, cause)
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

HTML = r"""<!doctype html><meta charset="utf-8"><title>DK Live Board</title>
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
<div id=wrap style="display:flex;gap:16px;align-items:flex-start">
<div id=grid></div>
<div id=right style="display:flex;flex-direction:column;gap:12px">
<aside id=mx style="font-family:ui-monospace,monospace;font-size:12px;
  line-height:1.35;min-width:210px;background:#161322;border:1px solid #2A2440;
  border-radius:4px;padding:10px 14px">
  <div id=runlabel style="font-weight:600;letter-spacing:.14em;color:#F2B33D">RUN &middot; SESSION</div>
  <div id=firstclear style="display:none;background:#E83C3C;color:#fff;
    font-weight:700;padding:4px 8px;border-radius:3px;margin:6px 0">&#9733; FIRST CLEAR &#9733;</div>
  <table id=mtable style="border-spacing:0;color:#E2DEEE"></table>
</aside>
<div id=deathslot></div>
</div>
</div>
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
// pro route (user doctrine + corridor): floor -> x203 -> g2 left ->
// wait spot -> x53 climb -> g3 right -> x131 -> g4 left -> x67 ->
// g5 right -> x147 -> top toward Pauline
// [x, y, extra-y-drop] — per-point tweak for visual girder alignment
const ROUTE=[[82,240,0],[203,236,0],[203,211,0],[59,202,0],[53,196,0],
  [53,178,0],[62,176,10],[131,158,16],[131,118,16],[94,122,26],
  [94,92,22],[200,97,20],[200,68,6],[120,65,6],[110,60,6],[110,35,6]];
for(let i=0;i<8;i++){
  const g=GROUPS[i];
  const cell=document.createElement('div'); cell.className='cell';
  cell.innerHTML='<img class=bg src="'+BG+'">'
    +'<svg viewBox="0 0 672 768" preserveAspectRatio="none"></svg>'
    +'<div class=tag>ENVS '+g[0]+'-'+g[g.length-1]+'</div>';
  grid.appendChild(cell);
  const sv=cell.querySelector('svg');
  const rp=document.createElementNS(NS,'polyline');
  rp.setAttribute('points',ROUTE.map(p=>((p[0]-14.5)*3)+','+((p[1]-7.5)*3+38+(p[2]||0))).join(' '));
  rp.setAttribute('fill','none');rp.setAttribute('stroke','#7BD88F');
  rp.setAttribute('stroke-width',7);rp.setAttribute('opacity',.28);
  rp.setAttribute('stroke-linejoin','round');rp.setAttribute('stroke-linecap','round');
  sv.appendChild(rp);
  panels.push(sv);
}
// death heatmap panel (in the right column, under the metrics)
{
  const cell=document.createElement('div'); cell.className='cell';
  cell.innerHTML='<img class=bg src="'+BG+'" style="filter:brightness(.45);width:260px">'
    +'<svg id=heat viewBox="0 0 672 768" preserveAspectRatio="none"></svg>'
    +'<div class=tag style="color:#E83C3C">DEATH MAP (session)</div>';
  document.getElementById('deathslot').appendChild(cell);
}
async function hpoll(){
  try{
    const ds=await (await fetch('/deaths')).json();
    const hs=document.getElementById('heat');
    while(hs.firstChild)hs.removeChild(hs.firstChild);
    for(const e of ds){
      const c=document.createElementNS(NS,'circle');
      const gy=240-e.h;
      // same warp as the route line: the bg frame's girders drift from a
      // linear RAM->pixel map as you go up the board, so interpolate the
      // extra drop from the user-calibrated ROUTE anchors (inverse-square
      // weights; floor anchors carry 0 so the bottom stays put).
      let wsum=0, osum=0;
      for(const p of ROUTE){
        const dx=e.x-p[0], dy=(gy-p[1])*2;
        const w=1/(dx*dx+dy*dy+25);
        wsum+=w; osum+=w*(p[2]||0);
      }
      c.setAttribute('cx',(e.x-14.5)*3);
      c.setAttribute('cy',(gy-7.5)*3+38+osum/wsum);
      c.setAttribute('r',e.glitch?5:3.5);
      c.setAttribute('fill',e.glitch?'#B26FD8':(e.hx?'#F2B33D':'#E83C3C'));
      c.setAttribute('opacity',.32);
      hs.appendChild(c);
    }
  }catch(err){}
  setTimeout(hpoll,7000);
}
hpoll();
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
async function mpoll(){
  try{
    const m=await (await fetch('/metrics')).json();
    document.getElementById('runlabel').textContent='RUN '+(m.label||'?').toUpperCase()+' \u00B7 SESSION';
    const rows=[
      ['honest clears', m.clears, m.clears>0],
      ['episodes', m.episodes, false],
      ['tower gates', m.gates, false],
      ['bottom-up mean h', m.bu_mean, false],
      ['bottom-up max h', m.bu_max, false],
      ['reached h65+', m.h65, false],
      ['passed waterfall (h68+)', m.h68, m.h68>0],
      ['wait-spot commit %', m.ws_rate+'%', false],
      ['wait-spot best gain', m.ws_best, false],
      ['floor crossing %', m.fl_rate+'%', false],
      ['floor best gain', m.fl_best, false],
      ['hammer pickups', m.hammer_pickups, false],
      ['deaths at hammer expiry', m.expiry_deaths, false],
      ['deaths bar/fball/self', (m.d_barrel||0)+'/'+(m.d_fireball||0)+'/'+(m.d_self||0), false],
      ['guard kills', m.guard_kills, false]];
    document.getElementById('mtable').innerHTML=rows.map(r=>
      '<tr><td style="color:#8B85A3;padding-right:12px">'+r[0]+'</td><td style="text-align:right;'+
      (r[2]?'color:#F2B33D;font-weight:700':'')+'">'+r[1]+'</td></tr>').join('');
    document.getElementById('firstclear').style.display=m.clears>0?'block':'none';
  }catch(e){}
  setTimeout(mpoll,5000);
}
poll(); mpoll(); requestAnimationFrame(render);
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
