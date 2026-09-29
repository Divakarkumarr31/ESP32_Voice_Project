from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import time
import wave
import socket
import struct
import threading
from collections import deque
from datetime import datetime

import numpy as np


# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)


# =====================================================
# WHISPER
# =====================================================

print("Loading Whisper model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded.")

whisper_lock = threading.Lock()


# =====================================================
# CONFIG — PS budget limits, shown on the dashboard
# =====================================================

RAM_BUDGET_KB = 256
CPU_BUDGET_PERCENT = 10
MODEL_SIZE_KB = 94  # update to your actual exported model size

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5002
STREAM_RECV_CHUNK = 2048
WAVEFORM_POINTS = 400


# =====================================================
# LATEST RESULT (kept, same as before)
# =====================================================

latest_text = "Waiting for ARISE..."
latest_ram_kb = "—"
latest_cpu_percent = "—"
latest_latency_ms = "—"


# =====================================================
# DETECTION LOG (powers the dashboard table + stats)
# =====================================================

detections = []
next_id = 1

device_connected = False
last_seen_ts = 0
DEVICE_TIMEOUT_SEC = 15

stream_active = False
waveform_lock = threading.Lock()
waveform_samples = deque([0.0] * WAVEFORM_POINTS, maxlen=WAVEFORM_POINTS)


def device_is_connected():
    return device_connected and (time.time() - last_seen_ts) < DEVICE_TIMEOUT_SEC


def mark_device_seen():
    global device_connected
    global last_seen_ts
    device_connected = True
    last_seen_ts = time.time()


def to_number(value):
    """Best-effort convert a header string to a float, else None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_wav_float32(path, target_rate=16000):
    """
    Load a WAV as float32 PCM for Faster Whisper.

    Avoids PyAV's av.open(..., metadata_errors=...), which crashes on
    current PyAV (19+) with: unexpected keyword argument 'metadata_errors'.
    """
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            "Expected 16-bit WAV, got sample width %s" % sample_width
        )

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    audio /= 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate and audio.size > 1:
        duration = audio.shape[0] / float(sample_rate)
        new_len = max(1, int(duration * target_rate))
        old_x = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        audio = np.interp(new_x, old_x, audio).astype(np.float32)

    return audio


def pcm_int16_to_float32(pcm_bytes):
    if len(pcm_bytes) < 2:
        return np.zeros(0, dtype=np.float32)

    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    audio = np.frombuffer(bytes(pcm_bytes[:usable]), dtype="<i2").astype(np.float32)
    audio /= 32768.0
    return audio


def push_waveform_pcm(pcm_bytes):
    """Append live samples to the dashboard rolling buffer (call per recv chunk)."""
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return

    samples = np.frombuffer(memoryview(pcm_bytes)[:usable], dtype="<i2")
    if samples.size == 0:
        return

    # One display point per ~10 ms so the canvas scrolls during speech.
    hop = 160
    points = []
    for i in range(0, samples.size, hop):
        chunk = samples[i:i + hop]
        peak = float(np.max(np.abs(chunk))) / 32768.0
        sign = 1.0 if int(chunk[-1]) >= 0 else -1.0
        points.append(sign * min(1.0, peak * 3.0))

    if not points:
        return

    with waveform_lock:
        waveform_samples.extend(points)


def transcribe_float32(audio):
    global latest_text

    if audio is None or audio.size < 1600:
        latest_text = "Could not recognize speech."
        return latest_text, None

    with whisper_lock:
        segments, info = model.transcribe(
            audio,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200
            },
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )

        text_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_parts.append(text)

        latest_text = " ".join(text_parts).strip()

    if not latest_text:
        latest_text = "Could not recognize speech."

    print()
    print("=" * 60)
    print("RECOGNIZED TEXT:")
    print("=" * 60)
    print(latest_text)
    print("=" * 60)
    print()

    if info is not None:
        print("Detected language:", info.language)
        print("Language probability:", info.language_probability)

    return latest_text, info


def log_detection(
    ram_kb,
    cpu_percent,
    stream_start_latency_ms,
    utterance_duration_ms,
    transcript
):
    global next_id
    global latest_ram_kb
    global latest_cpu_percent
    global latest_latency_ms

    latest_ram_kb = ram_kb
    latest_cpu_percent = cpu_percent
    latest_latency_ms = stream_start_latency_ms

    detections.append({
        "id": next_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "ram_kb": to_number(ram_kb) or 0,
        "cpu_percent": to_number(cpu_percent) or 0,
        "latency_ms": to_number(stream_start_latency_ms) or 0,
        "stream_start_latency_ms": to_number(stream_start_latency_ms) or 0,
        "utterance_duration_ms": to_number(utterance_duration_ms) or 0,
        "transcript": transcript,
        "status": "pending",
    })
    next_id += 1


def recv_exact(conn, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def handle_stream_client(conn, addr):
    global stream_active
    global latest_text

    stream_start_ts = time.time()
    first_byte_ts = None
    mark_device_seen()
    stream_active = True

    print()
    print("=" * 60)
    print("TCP STREAM FROM", addr)
    print("=" * 60)

    try:
        header = recv_exact(conn, 16)
        if header is None or len(header) < 16:
            print("Stream ended before format header.")
            return

        sample_rate, bits, channels, ram_kb, cpu_x10, stream_latency_ms = struct.unpack(
            "<IHHHHI",
            header
        )

        cpu_percent = cpu_x10 / 10.0

        print(
            "Format: %d Hz, %d-bit, %d ch | RAM %s KB | CPU %s%% | wake-to-stream %s ms"
            % (sample_rate, bits, channels, ram_kb, cpu_percent, stream_latency_ms)
        )

        pcm = bytearray()
        pending = bytearray()

        while True:
            data = conn.recv(STREAM_RECV_CHUNK)
            if not data:
                break

            if first_byte_ts is None:
                first_byte_ts = time.time()
                print(
                    "First audio byte after accept: %.1f ms"
                    % ((first_byte_ts - stream_start_ts) * 1000.0)
                )

            mark_device_seen()
            pending.extend(data)
            usable = len(pending) - (len(pending) % 2)
            if usable:
                chunk = bytes(pending[:usable])
                pcm.extend(chunk)
                push_waveform_pcm(chunk)
                del pending[:usable]

        utterance_duration_ms = 0
        if sample_rate > 0:
            utterance_duration_ms = round(
                (len(pcm) / 2) / float(sample_rate) * 1000.0,
                1
            )

        print("Stream closed. PCM bytes:", len(pcm))
        print("Utterance duration (informational):", utterance_duration_ms, "ms")
        print("Running Whisper speech recognition...")

        audio = pcm_int16_to_float32(pcm)
        text, info = transcribe_float32(audio)

        log_detection(
            ram_kb,
            cpu_percent,
            stream_latency_ms,
            utterance_duration_ms,
            text
        )

    except Exception as e:
        print("Stream handler error:", str(e))
        latest_text = "Speech recognition error: " + str(e)
        log_detection(0, 0, 0, 0, latest_text)

    finally:
        stream_active = False
        try:
            conn.close()
        except Exception:
            pass


def stream_server_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((STREAM_HOST, STREAM_PORT))
    sock.listen(4)
    print("Audio stream listener: tcp://0.0.0.0:%d" % STREAM_PORT)

    while True:
        conn, addr = sock.accept()
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        t = threading.Thread(
            target=handle_stream_client,
            args=(conn, addr),
            daemon=True
        )
        t.start()


def start_stream_server():
    t = threading.Thread(target=stream_server_loop, daemon=True)
    t.start()


# =====================================================
# DASHBOARD PAGE
# =====================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EdgeWake Dashboard</title>
<style>
:root {
  --bg: #0a0e14;
  --card-bg: #111827;
  --border: #1f2937;
  --accent: #22d3ee;
  --success: #4ade80;
  --danger: #f87171;
  --text-primary: #e5e7eb;
  --text-secondary: #9ca3af;
  --font-mono: 'Roboto Mono', 'Courier New', monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text-primary);
  font-family: -apple-system, Segoe UI, Arial, sans-serif;
  padding: 24px;
}
.wrap { max-width: 1100px; margin: 0 auto; }
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}
.logo { font-size: 20px; font-weight: 700; letter-spacing: 1px; }
.logo span { color: var(--accent); }
.status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); }
.dot.on { background: var(--success); }
.section-label {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--text-secondary);
  margin: 24px 0 8px;
}
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.card .label { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
.card .value { font-family: var(--font-mono); font-size: 22px; font-weight: 600; }
.card .sub { font-family: var(--font-mono); font-size: 13px; color: var(--text-secondary); }
.bar-track { height: 6px; background: var(--border); border-radius: 3px; margin-top: 10px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width .3s; }
.bar-fill.warn { background: var(--danger); }
.value.ok { color: var(--success); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--text-secondary); font-weight: 500; padding: 8px 6px; border-bottom: 1px solid var(--border); }
td { padding: 8px 6px; border-bottom: 1px solid var(--border); font-family: var(--font-mono); }
.tag { padding: 2px 8px; border-radius: 4px; font-size: 12px; font-family: -apple-system, sans-serif; }
.tag.pending { background: #1f2937; color: var(--text-secondary); }
.tag.tp { background: #052e1a; color: var(--success); }
.tag.fp { background: #3a0d0d; color: var(--danger); }
button.mark {
  font-size: 12px;
  padding: 3px 8px;
  margin-right: 4px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  border-radius: 4px;
  cursor: pointer;
}
button.mark:hover { border-color: var(--accent); color: var(--accent); }
.transcript {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  font-family: var(--font-mono);
  font-size: 14px;
  color: var(--text-secondary);
  min-height: 24px;
}
.empty { color: var(--text-secondary); font-size: 13px; padding: 12px 6px; }
#wave-canvas {
  width: 100%;
  height: 140px;
  display: block;
  background: #0b1220;
  border-radius: 6px;
}
</style>
</head>
<body>
<div class="wrap">

  <div class="header">
    <div class="logo">EDGE<span>WAKE</span></div>
    <div class="status">
      <div class="dot" id="conn-dot"></div>
      <span id="conn-text">Checking...</span>
    </div>
  </div>

  <div class="section-label">Live Audio Stream</div>
  <div class="card">
    <canvas id="wave-canvas" width="1000" height="140"></canvas>
    <div class="sub" id="wave-status" style="margin-top:10px">Waiting for stream...</div>
  </div>

  <div class="section-label">Efficiency</div>
  <div class="grid">
    <div class="card">
      <div class="label">RAM used</div>
      <div class="value" id="ram-value">-- / {{ ram_budget }} KB</div>
      <div class="bar-track"><div class="bar-fill" id="ram-bar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="label">Idle CPU</div>
      <div class="value" id="cpu-value">-- / {{ cpu_budget }}%</div>
      <div class="bar-track"><div class="bar-fill" id="cpu-bar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <div class="label">Model size</div>
      <div class="value">{{ model_size }} KB</div>
      <div class="sub">static, exported model</div>
    </div>
  </div>

  <div class="section-label">Accuracy</div>
  <div class="grid">
    <div class="card">
      <div class="label">True-positive rate</div>
      <div class="value ok" id="tp-rate">--</div>
    </div>
    <div class="card">
      <div class="label">False activations</div>
      <div class="value" id="fp-count">0</div>
    </div>
    <div class="card">
      <div class="label">Total detections</div>
      <div class="value" id="total-count">0</div>
    </div>
  </div>

  <div class="section-label">Timing</div>
  <div class="grid">
    <div class="card">
      <div class="label">Wake-to-stream latency</div>
      <div class="value" id="last-latency">-- ms</div>
      <div class="sub">keyword-end to TCP open</div>
    </div>
    <div class="card">
      <div class="label">Average wake-to-stream</div>
      <div class="value" id="avg-latency">-- ms</div>
    </div>
    <div class="card">
      <div class="label">Utterance duration</div>
      <div class="value" id="utterance-duration">-- ms</div>
      <div class="sub">not system latency</div>
    </div>
  </div>

  <div class="section-label">Live detections</div>
  <div class="card" style="padding:0">
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>RAM</th>
          <th>CPU</th>
          <th>Wake-to-stream</th>
          <th>Duration</th>
          <th>Transcript</th>
          <th>Result</th>
        </tr>
      </thead>
      <tbody id="log-body">
        <tr><td colspan="7" class="empty">Waiting for first detection...</td></tr>
      </tbody>
    </table>
  </div>

  <div class="section-label">Live ASR transcript</div>
  <div class="transcript" id="transcript-box">Waiting for ARISE...</div>

</div>

<script>
async function refresh() {
  try {
    const res = await fetch('/api/data');
    const data = await res.json();

    document.getElementById('conn-dot').className = 'dot' + (data.connected ? ' on' : '');
    document.getElementById('conn-text').textContent = data.connected ? 'Device connected' : 'Device disconnected';

    if (data.latest) {
      const ramPct = Math.min(100, (data.latest.ram_kb / {{ ram_budget }}) * 100);
      const cpuPct = Math.min(100, (data.latest.cpu_percent / {{ cpu_budget }}) * 100);
      const wakeMs = data.latest.stream_start_latency_ms != null ? data.latest.stream_start_latency_ms : data.latest.latency_ms;
      const durMs = data.latest.utterance_duration_ms != null ? data.latest.utterance_duration_ms : '--';

      document.getElementById('ram-value').textContent = data.latest.ram_kb + ' / {{ ram_budget }} KB';
      document.getElementById('ram-bar').style.width = ramPct + '%';
      document.getElementById('ram-bar').className = 'bar-fill' + (ramPct > 90 ? ' warn' : '');

      document.getElementById('cpu-value').textContent = data.latest.cpu_percent + ' / {{ cpu_budget }}%';
      document.getElementById('cpu-bar').style.width = cpuPct + '%';
      document.getElementById('cpu-bar').className = 'bar-fill' + (cpuPct > 90 ? ' warn' : '');

      document.getElementById('last-latency').textContent = wakeMs + ' ms';
      document.getElementById('utterance-duration').textContent = durMs + (durMs === '--' ? '' : ' ms');
      document.getElementById('transcript-box').textContent = data.latest.transcript || 'Could not recognize speech.';
    }

    document.getElementById('total-count').textContent = data.total;
    document.getElementById('fp-count').textContent = data.false_positives;
    document.getElementById('tp-rate').textContent = data.tp_rate === null ? '--' : data.tp_rate + '%';
    document.getElementById('avg-latency').textContent = data.avg_latency === null ? '-- ms' : data.avg_latency + ' ms';

    const tbody = document.getElementById('log-body');
    if (data.log.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty">Waiting for first detection...</td></tr>';
    } else {
      tbody.innerHTML = data.log.map(function(row) {
        let tag;
        if (row.status === 'true_positive') tag = '<span class="tag tp">True positive</span>';
        else if (row.status === 'false_activation') tag = '<span class="tag fp">False activation</span>';
        else tag = '<button class="mark" onclick="mark(' + row.id + ',\\'true_positive\\')">Correct</button><button class="mark" onclick="mark(' + row.id + ',\\'false_activation\\')">False</button>';

        const wake = row.stream_start_latency_ms != null ? row.stream_start_latency_ms : row.latency_ms;
        const dur = row.utterance_duration_ms != null ? row.utterance_duration_ms : '--';

        return '<tr>' +
          '<td>' + row.time + '</td>' +
          '<td>' + row.ram_kb + ' KB</td>' +
          '<td>' + row.cpu_percent + '%</td>' +
          '<td>' + wake + ' ms</td>' +
          '<td>' + dur + ' ms</td>' +
          '<td>' + (row.transcript ? row.transcript.slice(0, 40) : '--') + '</td>' +
          '<td>' + tag + '</td>' +
          '</tr>';
      }).join('');
    }
  } catch (e) {
    console.error(e);
  }
}

async function refreshWave() {
  try {
    const res = await fetch('/api/waveform');
    const data = await res.json();
    const canvas = document.getElementById('wave-canvas');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const samples = data.samples || [];

    document.getElementById('wave-status').textContent = data.streaming
      ? 'Streaming live PCM'
      : 'Idle — waiting for next wake';

    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1f2937';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const mid = h / 2;
    for (let i = 0; i < samples.length; i++) {
      const x = (i / Math.max(samples.length - 1, 1)) * w;
      const y = mid - (samples[i] * (h * 0.42));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  } catch (e) {
    console.error(e);
  }
}

async function mark(id, status) {
  await fetch('/api/mark/' + id + '/' + status, { method: 'POST' });
  refresh();
}

refresh();
refreshWave();
setInterval(refresh, 1500);
setInterval(refreshWave, 120);
</script>
</body>
</html>
"""


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template_string(
        DASHBOARD_HTML,
        ram_budget=RAM_BUDGET_KB,
        cpu_budget=CPU_BUDGET_PERCENT,
        model_size=MODEL_SIZE_KB
    )


# =====================================================
# DATA API — polled by the dashboard every 1.5s
# =====================================================

@app.route("/api/data")
def api_data():

    total = len(detections)
    false_positives = sum(1 for d in detections if d["status"] == "false_activation")
    marked = [d for d in detections if d["status"] in ("true_positive", "false_activation")]
    true_positives = sum(1 for d in marked if d["status"] == "true_positive")

    tp_rate = round((true_positives / len(marked)) * 100, 1) if marked else None

    latencies = [
        d.get("stream_start_latency_ms", d.get("latency_ms"))
        for d in detections
        if d.get("stream_start_latency_ms", d.get("latency_ms")) is not None
    ]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    latest = detections[-1] if detections else None

    return jsonify({
        "connected": device_is_connected() or stream_active,
        "total": total,
        "false_positives": false_positives,
        "tp_rate": tp_rate,
        "avg_latency": avg_latency,
        "latest": latest,
        "log": list(reversed(detections[-20:])),
    })


@app.route("/api/waveform")
def api_waveform():
    with waveform_lock:
        samples = list(waveform_samples)

    peak = max((abs(s) for s in samples), default=0.0)

    return jsonify({
        "samples": samples,
        "streaming": stream_active,
        "peak": round(peak, 4),
        "count": len(samples),
    })


@app.route("/api/mark/<int:detection_id>/<status>", methods=["POST"])
def api_mark(detection_id, status):

    if status not in ("true_positive", "false_activation"):
        return jsonify({"status": "error", "message": "invalid status"}), 400

    for d in detections:
        if d["id"] == detection_id:
            d["status"] = status
            return jsonify({"status": "ok"})

    return jsonify({"status": "error", "message": "not found"}), 404


# =====================================================
# AUDIO ENDPOINT (HTTP fallback — keep until stream is verified)
# =====================================================

@app.route(
    "/audio",
    methods=["POST"]
)
def receive_audio():

    global latest_text

    mark_device_seen()

    ram_kb = request.headers.get("X-Device-RAM-KB", "—")
    cpu_percent = request.headers.get("X-Device-CPU-Percent", "—")
    latency_ms = request.headers.get("X-Detection-Latency-MS", "—")

    print()
    print("=" * 60)
    print("AUDIO RECEIVED FROM ESP32 (HTTP FALLBACK)")
    print("=" * 60)

    print("Device RAM (free KB):", ram_kb)
    print("Device CPU (idle %):", cpu_percent)
    print("Detection latency (ms):", latency_ms)

    filename = "latest_audio.wav"

    with open(filename, "wb") as f:
        f.write(request.data)

    print("Audio saved:", filename)
    print("Audio size:", len(request.data), "bytes")
    print(
        "NOTE: HTTP /audio is fallback. Live waveform only moves on the "
        "TCP stream (port 5002). Restart simulate_esp32.py if it still "
        "prints Target server: http://127.0.0.1:5001/audio"
    )

    wav_payload = request.data
    if len(wav_payload) > 44:
        push_waveform_pcm(wav_payload[44:])

    print()
    print("Running Whisper speech recognition...")

    utterance_duration_ms = 0
    status_code = 200
    language = None

    try:
        audio = load_wav_float32(filename)
        utterance_duration_ms = round(audio.size / 16000.0 * 1000.0, 1)
        latest_text, info = transcribe_float32(audio)
        language = info.language if info is not None else None

        response_payload = {
            "status": "success",
            "text": latest_text,
            "language": language,
            "device_ram_kb": ram_kb,
            "device_cpu_percent": cpu_percent,
            "detection_latency_ms": latency_ms
        }

    except Exception as e:
        print("Whisper error:", str(e))
        latest_text = "Speech recognition error: " + str(e)
        response_payload = {
            "status": "error",
            "text": str(e)
        }
        status_code = 500

    log_detection(
        ram_kb,
        cpu_percent,
        latency_ms,
        utterance_duration_ms,
        latest_text
    )

    return jsonify(response_payload), status_code


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ARISE SERVER")
    print("=" * 60)
    print()
    print("Dashboard:")
    print("http://127.0.0.1:5001")
    print()
    print("Audio HTTP fallback:")
    print("http://127.0.0.1:5001/audio")
    print()
    print("Live PCM stream:")
    print("tcp://0.0.0.0:%d" % STREAM_PORT)
    print()
    print("Server starting...")
    print()

    start_stream_server()

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        threaded=True,
        use_reloader=False
    )
