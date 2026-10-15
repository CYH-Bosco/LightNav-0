"""Read-only browser display for the Jetson bench pipeline."""

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html><meta charset="utf-8"><title>LightNav Jetson preview</title>
<style>body{background:#101722;color:#eee;font:18px system-ui;margin:24px}
.views{display:flex;flex-wrap:wrap;gap:24px}img,canvas{width:min(640px,95vw);background:#172334}
pre{white-space:pre-wrap}small{color:#aab}</style>
<h1>LightNav trajectory preview</h1><p id="mode">Motion disabled · local velocity preview only</p>
<button id="stop" hidden onclick="fetch('/stop',{method:'POST'})">STOP ROBOT</button>
<div class="views"><div><p id="camera-note">Camera frame used for prediction</p>
<div style="position:relative;width:min(640px,95vw)"><img id="camera" style="display:block">
<canvas id="pointing" width="640" height="360" style="position:absolute;inset:0;width:100%;height:100%;background:transparent;pointer-events:none"></canvas></div>
<p id="pointing-note"></p></div>
<div><p>Trajectory: forward ↑ · left ←</p><canvas id="path" width="640" height="360"></canvas></div></div>
<pre id="status">Waiting for inference…</pre>
<small id="note">Robot remains stationary. Camera and path update on each inference response.</small>
<script>
const c=document.getElementById('path'),ctx=c.getContext('2d');
const pc=document.getElementById('pointing'),pctx=pc.getContext('2d');
function drawPointing(s){
 pctx.clearRect(0,0,640,360);
 for(const p of s.pointing_markers||[]){
  const x=p.x*640,y=p.y*360,color=p.channel==='apos'?'#80ffd6':'#ff60ac';
  pctx.strokeStyle='#101722';pctx.lineWidth=5;pctx.beginPath();pctx.arc(x,y,9,0,2*Math.PI);pctx.stroke();
  pctx.strokeStyle=color;pctx.lineWidth=3;pctx.stroke();
  pctx.beginPath();pctx.moveTo(x-15,y);pctx.lineTo(x+15,y);pctx.moveTo(x,y-15);pctx.lineTo(x,y+15);pctx.stroke();
  const label=p.channel==='apos'?'Affordance':'Goal object';pctx.font='bold 16px system-ui';
  const lx=Math.max(4,Math.min(x+18,636-pctx.measureText(label).width));
  const ly=Math.max(20,Math.min(y+(p.channel==='apos'?-18:28),350));
  pctx.lineWidth=4;pctx.strokeStyle='#101722';pctx.strokeText(label,lx,ly);
  pctx.fillStyle=color;pctx.fillText(label,lx,ly);
 }
 document.getElementById('pointing-note').textContent=s.seq===undefined?'':
  `Affordance: ${s.pointing?.apos_state||'not returned'} · Goal object: ${s.pointing?.opos_state||'not returned'}`;
}
function draw(points){
 ctx.clearRect(0,0,640,360);
 const range=Math.max(1,...points.flatMap(p=>[Math.abs(p[0]),Math.abs(p[1])]))*1.2;
 const scale=140/range,xy=p=>[320-p[1]*scale,190-p[0]*scale];
 ctx.strokeStyle='#34465f';ctx.lineWidth=1;
 ctx.beginPath();ctx.moveTo(320,10);ctx.lineTo(320,350);ctx.moveTo(10,190);ctx.lineTo(630,190);ctx.stroke();
 ctx.fillStyle='#aab';ctx.font='14px system-ui';ctx.fillText('Forward',330,22);
 ctx.fillText('Left',12,180);ctx.fillText('Robot',330,210);ctx.fillText('±'+range.toFixed(2)+' m',12,340);
 ctx.strokeStyle='#49bcff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(320,190);
 for(const p of points){const [x,y]=xy(p);ctx.lineTo(x,y);}ctx.stroke();
 for(const p of points){const [x,y]=xy(p);ctx.fillStyle='#49bcff';ctx.beginPath();ctx.arc(x,y,4,0,7);ctx.fill();
 ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-14*Math.sin(p[2]),y-14*Math.cos(p[2]));ctx.stroke();}
 ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(320,190,5,0,7);ctx.fill();
}
let sequence=null;
async function update(){try{
 const r=await fetch('/state.json',{cache:'no-store'});const s=await r.json();
 drawPointing(s);
 const live=s.live_camera||s.motion_enabled!==undefined;
 document.getElementById('mode').textContent=s.motion_enabled?'REAL MOTION ENABLED':'Motion disabled · local velocity preview only';
 document.getElementById('stop').hidden=!s.motion_enabled;
 if(live){document.getElementById('camera').src='/camera.jpg?t='+Date.now();
 document.getElementById('camera-note').textContent='Live ego RGB · points from latest inference';
 document.getElementById('note').textContent='Trajectory follows measured wheel odometry. Ctrl+C or STOP ends the run.';}
 if(s.seq!==undefined){
  if(live||sequence!==s.seq){sequence=s.seq;if(!live)document.getElementById('camera').src='/camera.jpg?seq='+s.seq;draw(s.waypoints);}
  const v=s.velocity_preview;
  document.getElementById('status').textContent=`Sequence ${s.seq} · ${s.waypoints.length} waypoints · ${s.latency_ms.toFixed(1)} ms\n`+
  `Forward ${v.linear_x_m_s.toFixed(3)} m/s · Yaw ${v.angular_z_rad_s.toFixed(3)} rad/s · Stop ${s.stop}\n`+
  (s.motion_enabled?'Commanded limit '+s.max_vel+' m/s · Measured '+s.measured_forward_m_s.toFixed(3)+' m/s':'No motion published');
 }else if(s.status){
  sequence=null;draw([]);if(!live)document.getElementById('camera').removeAttribute('src');
  document.getElementById('status').textContent=s.status+' · no valid trajectory or velocity\n'+(s.error||'');
 }
}catch(e){drawPointing({});draw([]);document.getElementById('status').textContent='Preview disconnected';}
setTimeout(update,100);}draw([]);update();
</script>"""


def pointing_record(result, width, height):
    """Keep inference pixels and normalized markers for any browser image size."""
    pointing = {"frame_size": [width, height]}
    markers = []
    for channel in ("apos", "opos"):
        state = getattr(result, f"{channel}_state", None)
        pixel = getattr(result, f"{channel}_px", None)
        pointing[f"{channel}_state"] = state
        pointing[f"{channel}_px"] = pixel
        if state != "point" or pixel is None:
            continue
        x, y = pixel
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height:
            markers.append({"channel": channel, "x": x / width, "y": y / height})
    return {"pointing": pointing, "pointing_markers": markers}


class PreviewServer:
    def __init__(self, host: str, port: int, on_stop=None):
        self._lock = threading.Lock()
        self._state = b'{}'
        self._jpeg = b''
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                with owner._lock:
                    if path == "/":
                        body, content_type = PAGE.encode(), "text/html; charset=utf-8"
                    elif path == "/state.json":
                        body, content_type = owner._state, "application/json"
                    elif path == "/camera.jpg" and owner._jpeg:
                        body, content_type = owner._jpeg, "image/jpeg"
                    else:
                        self.send_error(404)
                        return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != "/stop" or on_stop is None:
                    self.send_error(501)
                    return
                on_stop()
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def publish(self, rgb, record):
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("could not encode camera preview")
        state = json.dumps(record, allow_nan=False).encode()
        with self._lock:
            self._jpeg = jpeg.tobytes()
            self._state = state

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self._thread.join()

    def publish_status(self, event, clear_camera=True):
        state = json.dumps(event, allow_nan=False).encode()
        with self._lock:
            if clear_camera:
                self._jpeg = b""
            self._state = state

    def publish_camera(self, rgb):
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("could not encode camera preview")
        with self._lock:
            self._jpeg = jpeg.tobytes()

    def publish_record(self, record):
        state = json.dumps(record, allow_nan=False).encode()
        with self._lock:
            self._state = state
