from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import os
import time
import wave
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


# =====================================================
# CONFIG — PS budget limits, shown on the dashboard
# =====================================================

RAM_BUDGET_KB = 256
CPU_BUDGET_PERCENT = 10
MODEL_SIZE_KB = 94  # update to your actual exported model size


# =====================================================
# LATEST RESULT (kept, same as before)
# =====================================================

latest_text = "Waiting for ARISE..."
latest_ram_kb = "—"
latest_cpu_percent = "—"
latest_latency_ms = "—"


# =====================================================
# DETECTION LOG (new — powers the dashboard table + stats)
# =====================================================

detections = []
next_id = 1

device_connected = False
last_seen_ts = 0
DEVICE_TIMEOUT_SEC = 15


def device_is_connected():
    return device_connected and (time.time() - last_seen_ts) < DEVICE_TIMEOUT_SEC


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

  <div class="section-label">Latency</div>
  <div class="grid">
    <div class="card">
      <div class="label">Last detection</div>
      <div class="value" id="last-latency">-- ms</div>
    </div>
    <div class="card">
      <div class="label">Average latency</div>
      <div class="value" id="avg-latency">-- ms</div>
    </div>
  </div>

  <div class="section-label">Live detections</div>
  <div class="card" style="padding:0">
    <table>
      <thead>
        <tr><th>Time</th><th>RAM</th><th>CPU</th><th>Latency</th><th>Transcript</th><th>Result</th></tr>
      </thead>
      <tbody id="log-body">
        <tr><td colspan="6" class="empty">Waiting for first detection...</td></tr>
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

      document.getElementById('ram-value').textContent = data.latest.ram_kb + ' / {{ ram_budget }} KB';
      document.getElementById('ram-bar').style.width = ramPct + '%';
      document.getElementById('ram-bar').className = 'bar-fill' + (ramPct > 90 ? ' warn' : '');

      document.getElementById('cpu-value').textContent = data.latest.cpu_percent + ' / {{ cpu_budget }}%';
      document.getElementById('cpu-bar').style.width = cpuPct + '%';
      document.getElementById('cpu-bar').className = 'bar-fill' + (cpuPct > 90 ? ' warn' : '');

      document.getElementById('last-latency').textContent = data.latest.latency_ms + ' ms';
      document.getElementById('transcript-box').textContent = data.latest.transcript || 'Could not recognize speech.';
    }

    document.getElementById('total-count').textContent = data.total;
    document.getElementById('fp-count').textContent = data.false_positives;
    document.getElementById('tp-rate').textContent = data.tp_rate === null ? '--' : data.tp_rate + '%';
    document.getElementById('avg-latency').textContent = data.avg_latency === null ? '-- ms' : data.avg_latency + ' ms';

    const tbody = document.getElementById('log-body');
    if (data.log.length === 0) {
      tbody.innerHTML = '<tr><td colspan="6" class="empty">Waiting for first detection...</td></tr>';
    } else {
      tbody.innerHTML = data.log.map(function(row) {
        let tag;
        if (row.status === 'true_positive') tag = '<span class="tag tp">True positive</span>';
        else if (row.status === 'false_activation') tag = '<span class="tag fp">False activation</span>';
        else tag = '<button class="mark" onclick="mark(' + row.id + ',\\'true_positive\\')">Correct</button><button class="mark" onclick="mark(' + row.id + ',\\'false_activation\\')">False</button>';

        return '<tr>' +
          '<td>' + row.time + '</td>' +
          '<td>' + row.ram_kb + ' KB</td>' +
          '<td>' + row.cpu_percent + '%</td>' +
          '<td>' + row.latency_ms + ' ms</td>' +
          '<td>' + (row.transcript ? row.transcript.slice(0, 40) : '--') + '</td>' +
          '<td>' + tag + '</td>' +
          '</tr>';
      }).join('');
    }
  } catch (e) {
    console.error(e);
  }
}

async function mark(id, status) {
  await fetch('/api/mark/' + id + '/' + status, { method: 'POST' });
  refresh();
}

refresh();
setInterval(refresh, 1500);
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

    latencies = [d["latency_ms"] for d in detections if d["latency_ms"] is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    latest = detections[-1] if detections else None

    return jsonify({
        "connected": device_is_connected(),
        "total": total,
        "false_positives": false_positives,
        "tp_rate": tp_rate,
        "avg_latency": avg_latency,
        "latest": latest,
        "log": list(reversed(detections[-20:])),
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
# AUDIO ENDPOINT
# =====================================================

@app.route(
    "/audio",
    methods=["POST"]
)
def receive_audio():

    global latest_text
    global latest_ram_kb
    global latest_cpu_percent
    global latest_latency_ms
    global next_id
    global device_connected
    global last_seen_ts

    device_connected = True
    last_seen_ts = time.time()

    latest_ram_kb = request.headers.get(
        "X-Device-RAM-KB",
        "—"
    )

    latest_cpu_percent = request.headers.get(
        "X-Device-CPU-Percent",
        "—"
    )

    latest_latency_ms = request.headers.get(
        "X-Detection-Latency-MS",
        "—"
    )

    print()
    print("=" * 60)
    print("AUDIO RECEIVED FROM ESP32")
    print("=" * 60)

    print(
        "Device RAM (free KB):",
        latest_ram_kb
    )

    print(
        "Device CPU (idle %):",
        latest_cpu_percent
    )

    print(
        "Detection latency (ms):",
        latest_latency_ms
    )


    # =================================================
    # SAVE WAV
    # =================================================

    filename = "latest_audio.wav"


    with open(
        filename,
        "wb"
    ) as f:

        f.write(
            request.data
        )


    print(
        "Audio saved:",
        filename
    )


    print(
        "Audio size:",
        len(request.data),
        "bytes"
    )


    # =================================================
    # WHISPER
    # =================================================

    print()
    print("Running Whisper speech recognition...")


    try:

        audio = load_wav_float32(filename)

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


        latest_text = " ".join(
            text_parts
        ).strip()


        if not latest_text:

            latest_text = "Could not recognize speech."


        print()
        print("=" * 60)

        print("RECOGNIZED TEXT:")

        print("=" * 60)

        print(latest_text)

        print("=" * 60)

        print()

        print(
            "Detected language:",
            info.language
        )

        print(
            "Language probability:",
            info.language_probability
        )

        response_payload = {

            "status": "success",

            "text": latest_text,

            "language": info.language,

            "device_ram_kb": latest_ram_kb,

            "device_cpu_percent": latest_cpu_percent,

            "detection_latency_ms": latest_latency_ms

        }
        status_code = 200


    except Exception as e:

        print(
            "Whisper error:",
            str(e)
        )


        latest_text = (
            "Speech recognition error: "
            + str(e)
        )

        response_payload = {

            "status": "error",

            "text": str(e)

        }
        status_code = 500

    # =================================================
    # LOG DETECTION FOR DASHBOARD
    # =================================================

    detections.append({
        "id": next_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "ram_kb": to_number(latest_ram_kb) or 0,
        "cpu_percent": to_number(latest_cpu_percent) or 0,
        "latency_ms": to_number(latest_latency_ms) or 0,
        "transcript": latest_text,
        "status": "pending",
    })
    next_id += 1

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

    print(
        "http://127.0.0.1:5001"
    )

    print()

    print("Audio endpoint:")

    print(
        "http://127.0.0.1:5001/audio"
    )

    print()

    print("Server starting...")

    print()


    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )