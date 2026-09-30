from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import time
import wave
import socket
import struct
import threading
import io
import base64
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image


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
WAKE_THRESHOLD = 0.80  # must match firmware WAKE_THRESHOLD

# Team / event details shown in the dashboard navbar
EVENT_NAME = "SIH 2026"
TEAM_NAME = "Neural Nomads1"
TEAM_ID = "183857"
PS_ID = "26172"

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 5002
STREAM_RECV_CHUNK = 2048
WAVEFORM_POINTS = 400
STREAM_TAG_WAKE = 0x01
STREAM_TAG_COMMAND = 0x02


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
first_connected_ts = None
disconnect_count = 0
was_connected = False

stream_active = False
waveform_lock = threading.Lock()
waveform_samples = deque([0.0] * WAVEFORM_POINTS, maxlen=WAVEFORM_POINTS)
latest_wake_window = None

# Pipeline state machine for the "Live Processing" panel.
# listening -> wake_detected -> streaming -> transcribing -> listening
PIPELINE_STATE_STALE_SEC = 12
current_pipeline_state = "listening"
pipeline_state_updated_ts = time.time()


def set_pipeline_state(state):
    global current_pipeline_state
    global pipeline_state_updated_ts
    current_pipeline_state = state
    pipeline_state_updated_ts = time.time()


def get_pipeline_state():
    """Read-time safety net: fall back to 'listening' if a state got
    stuck (e.g. a handler thread died mid-stream) instead of showing a
    permanently wrong badge on the dashboard."""
    if current_pipeline_state != "listening":
        if (time.time() - pipeline_state_updated_ts) > PIPELINE_STATE_STALE_SEC:
            return "listening"
    return current_pipeline_state


def device_is_connected():
    return device_connected and (time.time() - last_seen_ts) < DEVICE_TIMEOUT_SEC


def mark_device_seen():
    global device_connected
    global last_seen_ts
    global first_connected_ts
    device_connected = True
    last_seen_ts = time.time()
    if first_connected_ts is None:
        first_connected_ts = time.time()


def update_disconnect_count(currently_connected):
    """Increment only on a True -> False transition, not every poll."""
    global was_connected
    global disconnect_count
    if was_connected and not currently_connected:
        disconnect_count += 1
    was_connected = currently_connected


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


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft, n_mels, sample_rate, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0

    n_bins = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (
                np.arange(left, center) - left
            ) / float(center - left)
        if right > center:
            fb[i, center:right] = (
                right - np.arange(center, right)
            ) / float(right - center)
    return fb


def compute_mel_spectrogram_png(int16_pcm, sample_rate=16000):
    """Return a compact log-mel PNG data URI, or None if audio is empty."""
    if int16_pcm is None:
        return None

    pcm = np.asarray(int16_pcm, dtype=np.int16).reshape(-1)
    if pcm.size < 256:
        return None

    audio = pcm.astype(np.float32) / 32768.0
    n_fft = 512
    hop = 160
    n_mels = 40
    window = np.hanning(n_fft).astype(np.float32)

    frames = []
    if audio.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[:audio.size] = audio
        frames.append(padded * window)
    else:
        for start in range(0, audio.size - n_fft + 1, hop):
            frames.append(audio[start:start + n_fft] * window)

    spec = np.stack(frames, axis=1)
    power = np.abs(np.fft.rfft(spec, n=n_fft, axis=0)) ** 2
    fb = _mel_filterbank(n_fft, n_mels, sample_rate)
    mel = np.dot(fb, power)
    log_mel = 10.0 * np.log10(mel + 1e-10)

    lo = np.percentile(log_mel, 5)
    hi = np.percentile(log_mel, 99)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((log_mel - lo) / (hi - lo), 0.0, 1.0)
    norm = np.flipud(norm)

    r = np.clip(1.4 * norm, 0, 1)
    g = np.clip(1.6 * norm - 0.35, 0, 1)
    b = np.clip(1.15 - 0.85 * norm, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((320, 120), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


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
    transcript,
    confidence=None,
    command_pcm=None
):
    global next_id
    global latest_ram_kb
    global latest_cpu_percent
    global latest_latency_ms
    global latest_wake_window

    latest_ram_kb = ram_kb
    latest_cpu_percent = cpu_percent
    latest_latency_ms = stream_start_latency_ms

    ram_n = to_number(ram_kb) or 0
    cpu_n = to_number(cpu_percent) or 0
    conf_n = to_number(confidence)
    within_budget = (ram_n <= RAM_BUDGET_KB) and (cpu_n <= CPU_BUDGET_PERCENT)

    wake_png = compute_mel_spectrogram_png(latest_wake_window)
    command_png = compute_mel_spectrogram_png(command_pcm)

    detections.append({
        "id": next_id,
        "time": datetime.now().strftime("%H:%M:%S"),
        "ram_kb": ram_n,
        "cpu_percent": cpu_n,
        "latency_ms": to_number(stream_start_latency_ms) or 0,
        "stream_start_latency_ms": to_number(stream_start_latency_ms) or 0,
        "utterance_duration_ms": to_number(utterance_duration_ms) or 0,
        "confidence": round(conf_n, 2) if conf_n is not None else None,
        "within_budget": within_budget,
        "wake_spectrogram_png": wake_png,
        "command_spectrogram_png": command_png,
        "wake_window_samples": int(np.asarray(latest_wake_window).size) if latest_wake_window is not None else 0,
        "transcript": transcript,
        "status": "pending",
    })
    next_id += 1
    latest_wake_window = None


def recv_exact(conn, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def handle_wake_window(conn):
    global latest_wake_window

    count_bytes = recv_exact(conn, 4)
    if count_bytes is None:
        print("Wake window missing sample count.")
        return

    n_samples = struct.unpack("<I", count_bytes)[0]
    raw = recv_exact(conn, n_samples * 2)
    if raw is None:
        print("Wake window truncated. expected samples:", n_samples)
        return

    latest_wake_window = np.frombuffer(raw, dtype="<i2").copy()
    set_pipeline_state("wake_detected")
    print()
    print("=" * 60)
    print("WAKE WINDOW RECEIVED")
    print("=" * 60)
    print("Samples:", latest_wake_window.size)
    print("Duration ms:", round(latest_wake_window.size / 16000.0 * 1000.0, 1))


def handle_command_stream(conn, addr):
    global stream_active
    global latest_text

    stream_start_ts = time.time()
    first_byte_ts = None
    stream_active = True

    print()
    print("=" * 60)
    print("COMMAND STREAM FROM", addr)
    print("=" * 60)

    header = recv_exact(conn, 18)
    if header is None or len(header) < 18:
        print("Stream ended before format header.")
        return

    (
        sample_rate,
        bits,
        channels,
        ram_kb,
        cpu_x10,
        stream_latency_ms,
        confidence_x100,
    ) = struct.unpack("<IHHHHIH", header)

    cpu_percent = cpu_x10 / 10.0
    confidence = confidence_x100 / 100.0
    set_pipeline_state("streaming")

    print(
        "Format: %d Hz, %d-bit, %d ch | RAM %s KB | CPU %s%% | "
        "wake-to-stream %s ms | confidence %.2f%%"
        % (
            sample_rate,
            bits,
            channels,
            ram_kb,
            cpu_percent,
            stream_latency_ms,
            confidence,
        )
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

    set_pipeline_state("transcribing")
    audio = pcm_int16_to_float32(pcm)
    text, info = transcribe_float32(audio)
    command_pcm = np.frombuffer(bytes(pcm), dtype="<i2").copy() if len(pcm) >= 2 else None

    log_detection(
        ram_kb,
        cpu_percent,
        stream_latency_ms,
        utterance_duration_ms,
        text,
        confidence,
        command_pcm
    )
    set_pipeline_state("listening")


def handle_stream_client(conn, addr):
    global stream_active
    global latest_text

    mark_device_seen()

    try:
        tag_b = recv_exact(conn, 1)
        if tag_b is None:
            print("Empty TCP connection.")
            return

        tag = tag_b[0]

        if tag == STREAM_TAG_WAKE:
            handle_wake_window(conn)
        elif tag == STREAM_TAG_COMMAND:
            handle_command_stream(conn, addr)
        else:
            print("Unknown stream tag:", tag)

    except Exception as e:
        print("Stream handler error:", str(e))
        latest_text = "Speech recognition error: " + str(e)
        log_detection(0, 0, 0, 0, latest_text, None, None)

    finally:
        stream_active = False
        set_pipeline_state("listening")
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #080c12;
  --panel: #0e131c;
  --inset: #0a0e15;
  --border: #1c2534;
  --border-strong: #283447;
  --accent: #22d3ee;
  --success: #4ade80;
  --warn: #fbbf24;
  --danger: #f87171;
  --text: #e7ebf1;
  --text-dim: #7c8aa0;
  --text-faint: #56647a;
  --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, 'Courier New', monospace;
  --sans: 'Inter', -apple-system, 'Segoe UI', Arial, sans-serif;
  --s1: 8px;
  --s2: 12px;
  --s3: 16px;
  --radius: 8px;
}
* { box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}
.board {
  height: 100vh;
  padding: var(--s2) var(--s3);
  display: grid;
  grid-template-rows: auto minmax(0, 19vh) minmax(0, 25vh) minmax(0, 1fr);
  gap: var(--s2);
}

/* ---------- header ---------- */
.topbar {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  column-gap: 28px;
  padding: 2px 2px 4px;
}
.brand .logo {
  font-family: var(--mono);
  font-size: clamp(2rem, 3.1vw, 2.7rem);
  font-weight: 700;
  letter-spacing: 2px;
  line-height: 1;
}
.brand .logo em { font-style: normal; color: var(--accent); }
.brand .tagline { font-size: 0.78rem; color: var(--text-dim); margin-top: 6px; letter-spacing: 0.2px; }
.meta-strip {
  justify-self: center;
  display: flex;
  align-items: stretch;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.meta-item { padding: 8px 18px; border-left: 1px solid var(--border); min-width: 0; }
.meta-item:first-child { border-left: none; }
.meta-item .label {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 1.1px;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: 5px;
  white-space: nowrap;
}
.meta-item .value {
  font-family: var(--mono);
  font-size: 0.95rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--text);
  white-space: nowrap;
}
.meta-item .value.accent { color: var(--accent); }
.status { text-align: right; font-size: 0.7rem; color: var(--text-dim); line-height: 1.5; }
.status .conn-line {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 6px;
  font-family: var(--mono);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  color: var(--text);
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--danger); flex-shrink: 0; }
.dot.on { background: var(--success); }

/* ---------- panels ---------- */
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: var(--s2) var(--s3);
  display: flex;
  flex-direction: column;
  min-height: 0;
  min-width: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: var(--s2);
  flex-shrink: 0;
}
.panel-title {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 1.1px;
  text-transform: uppercase;
  color: var(--text-dim);
  border-left: 2px solid var(--accent);
  padding-left: var(--s1);
  line-height: 1;
}

/* ---------- shared stat anatomy ---------- */
.stat { min-width: 0; display: flex; flex-direction: column; justify-content: flex-end; }
.stat .label {
  font-size: 0.72rem;
  font-weight: 500;
  color: var(--text-dim);
  white-space: nowrap;
  margin-bottom: 6px;
}
.stat .value, .ce-metric .value {
  font-family: var(--mono);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 1;
  white-space: nowrap;
  color: var(--text);
}
.stat .value.hero { font-size: clamp(1.9rem, 3.1vw, 2.6rem); letter-spacing: -0.5px; }
.stat .value.sec { font-size: clamp(1.1rem, 1.7vw, 1.4rem); }
.value.primary { color: var(--success); }
.unit {
  font-family: var(--sans);
  font-size: 0.42em;
  font-weight: 500;
  letter-spacing: 0;
  color: var(--text-dim);
  margin-left: 4px;
}
.stat .value.sec .unit { font-size: 0.55em; }
.bar-track { height: 4px; background: var(--border); border-radius: 2px; margin-top: var(--s1); overflow: hidden; }
.bar-track.spacer { background: transparent; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 2px; }
.bar-fill.warn { background: var(--danger); }

.badge {
  display: inline-block;
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.3px;
  padding: 2px 8px;
  border-radius: 4px;
  white-space: nowrap;
}
.badge.ok { background: #052e1a; color: var(--success); }
.badge.bad { background: #3a0d0d; color: var(--danger); }

/* ---------- top row ---------- */
.evaluation {
  display: grid;
  grid-template-columns: 1.3fr 1fr 0.8fr;
  gap: var(--s2);
  min-height: 0;
}
.eval-panel { border-color: var(--border-strong); }
.eval-body { flex: 1; min-height: 0; display: grid; align-items: end; }
.resources-panel .eval-body { grid-template-columns: repeat(3, 1fr); }
.detect-panel .eval-body { grid-template-columns: 1.25fr 1fr; }
.latency-panel .eval-body { grid-template-columns: 1.25fr 1fr; }
.eval-body > .stat { padding: 0 var(--s3); border-left: 1px solid var(--border); }
.eval-body > .stat:first-child { padding-left: 0; border-left: none; }
.eval-body > .stat:last-child { padding-right: 0; }
.stat-stack { display: flex; flex-direction: column; gap: var(--s2); padding-left: var(--s3); border-left: 1px solid var(--border); }
.stat-stack .stat { justify-content: flex-start; }

/* ---------- live processing ---------- */
.state-badge {
  font-family: var(--mono);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  padding: 3px 10px;
  border-radius: 4px;
  border: 1px solid var(--border-strong);
  color: var(--text-dim);
}
.state-badge.wake_detected { color: var(--warn); border-color: var(--warn); }
.state-badge.streaming { color: var(--accent); border-color: var(--accent); }
.state-badge.transcribing { color: var(--success); border-color: var(--success); }

.pipeline-body {
  display: grid;
  grid-template-columns: 1.6fr 1fr 1fr;
  gap: var(--s2);
  flex: 1;
  min-height: 0;
}
.pipe-cell { display: flex; flex-direction: column; min-height: 0; min-width: 0; }
.pipe-cell .cell-label {
  font-size: 0.66rem;
  font-weight: 500;
  color: var(--text-faint);
  margin-bottom: 6px;
  flex-shrink: 0;
}
#wave-canvas {
  width: 100%;
  flex: 1;
  min-height: 0;
  display: block;
  background: var(--inset);
  border: 1px solid var(--border);
  border-radius: 6px;
}
.spec-body {
  flex: 1;
  min-height: 0;
  border-radius: 6px;
  border: 1px solid var(--border);
  overflow: hidden;
  background: var(--inset);
  position: relative;
}
.spec-body img { width: 100%; height: 100%; object-fit: cover; display: block; }
.spec-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-faint);
  font-size: 0.68rem;
  text-align: center;
  padding: var(--s1);
}

/* ---------- bottom row ---------- */
.bottom {
  display: grid;
  grid-template-columns: 37fr 63fr;
  gap: var(--s2);
  min-height: 0;
}

/* latest detection card */
.latest-card { gap: var(--s2); }
.ce-top { display: flex; flex-direction: column; gap: 6px; }
.ce-status-line {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 0.66rem;
  font-weight: 700;
  letter-spacing: 0.8px;
  color: var(--text-faint);
}
.ce-status-line .dot { width: 7px; height: 7px; background: var(--text-faint); }
.ce-status-line.active { color: var(--success); }
.ce-status-line.active .dot { background: var(--success); }
.ce-keyword {
  font-family: var(--mono);
  font-size: clamp(1.7rem, 2.8vw, 2.4rem);
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--accent);
  line-height: 1;
}
.ce-metrics {
  display: grid;
  grid-template-columns: 1fr 1fr;
  padding: var(--s2) 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.ce-metric:last-child { padding-left: var(--s3); border-left: 1px solid var(--border); }
.ce-metric .label { font-size: 0.68rem; color: var(--text-dim); margin-bottom: 6px; }
.ce-metric .value { font-size: clamp(1.2rem, 1.9vw, 1.6rem); }
.value.conf-ok { color: var(--success); }
.value.conf-warn { color: var(--warn); }
.value.conf-bad { color: var(--danger); }
.ce-transcript-col {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: var(--inset);
  border: 1px solid var(--border);
  border-left: 2px solid var(--accent);
  border-radius: 6px;
  padding: var(--s2) var(--s3);
}
.ce-transcript-col .label {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 1.1px;
  text-transform: uppercase;
  color: var(--text-faint);
  margin-bottom: var(--s1);
  flex-shrink: 0;
}
.ce-transcript-col .text {
  flex: 1;
  min-height: 0;
  font-size: clamp(1.05rem, 1.6vw, 1.45rem);
  font-weight: 500;
  line-height: 1.4;
  color: var(--text);
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
  word-break: break-word;
}

/* history table */
.table-wrap { flex: 1; min-height: 0; overflow-y: auto; border-radius: 6px; }
table { width: 100%; border-collapse: collapse; font-size: 0.72rem; }
thead th {
  position: sticky;
  top: 0;
  background: var(--panel);
  text-align: left;
  color: var(--text-faint);
  font-weight: 600;
  font-size: 0.62rem;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  padding: 6px 8px;
  border-bottom: 1px solid var(--border-strong);
  white-space: nowrap;
}
td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  color: var(--text-dim);
  white-space: nowrap;
}
tbody tr:hover td { background: rgba(255,255,255,0.02); }
th:nth-child(2), th:nth-child(3), th:nth-child(5), th:nth-child(6),
td:nth-child(2), td:nth-child(3), td:nth-child(5), td:nth-child(6) { text-align: right; }
td:first-child { color: var(--text); }
td.transcript-cell {
  font-family: var(--sans);
  color: var(--text);
  max-width: 14vw;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tag {
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.3px;
  font-family: var(--mono);
  white-space: nowrap;
  display: inline-block;
}
.tag.pending { background: #1a2130; color: var(--text-dim); }
.tag.tp { background: #052e1a; color: var(--success); }
.tag.fp { background: #3a0d0d; color: var(--danger); }
.conf-ok { color: var(--success); }
.conf-warn { color: var(--warn); }
.conf-bad { color: var(--danger); }
button.mark {
  font-family: var(--mono);
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.2px;
  padding: 2px 7px;
  margin-left: 4px;
  background: transparent;
  border: 1px solid var(--border-strong);
  color: var(--text-dim);
  border-radius: 4px;
  cursor: pointer;
}
button.mark.correct:hover { border-color: var(--success); color: var(--success); }
button.mark.wrong:hover { border-color: var(--danger); color: var(--danger); }
.empty { color: var(--text-faint); font-size: 0.75rem; padding: var(--s2) 8px; font-family: var(--sans); }
</style>
</head>
<body>
<div class="board">

  <div class="topbar">
    <div class="brand">
      <div class="logo">EDGE<em>WAKE</em></div>
      <div class="tagline">On-device wake-word detection</div>
    </div>
    <div class="meta-strip">
      <div class="meta-item"><div class="label">Event</div><div class="value accent">{{ event_name }}</div></div>
      <div class="meta-item"><div class="label">Team</div><div class="value">{{ team_name }}</div></div>
      <div class="meta-item"><div class="label">Team ID</div><div class="value">{{ team_id }}</div></div>
      <div class="meta-item"><div class="label">Problem statement</div><div class="value">{{ ps_id }}</div></div>
    </div>
    <div class="status">
      <div class="conn-line"><div class="dot" id="conn-dot"></div><span id="conn-text">CHECKING...</span></div>
      <div id="status-meta"></div>
    </div>
  </div>

  <!-- 1. EVALUATION -->
  <div class="evaluation">
    <div class="panel eval-panel resources-panel">
      <div class="panel-head"><div class="panel-title">System resources</div></div>
      <div class="eval-body">
        <div class="stat">
          <div class="label">RAM</div>
          <div class="value hero" id="ram-value"><span class="metric-num">--</span><span class="unit">/ {{ ram_budget }} KB</span></div>
          <div class="bar-track"><div class="bar-fill" id="ram-bar" style="width:0%"></div></div>
        </div>
        <div class="stat">
          <div class="label">CPU</div>
          <div class="value hero" id="cpu-value"><span class="metric-num">--</span><span class="unit">/ {{ cpu_budget }}%</span></div>
          <div class="bar-track"><div class="bar-fill" id="cpu-bar" style="width:0%"></div></div>
        </div>
        <div class="stat">
          <div class="label">Model size</div>
          <div class="value hero"><span class="metric-num">{{ model_size }}</span><span class="unit">KB</span></div>
          <div class="bar-track spacer"></div>
        </div>
      </div>
    </div>
    <div class="panel eval-panel detect-panel">
      <div class="panel-head"><div class="panel-title">Detection</div></div>
      <div class="eval-body">
        <div class="stat">
          <div class="label">True-positive rate</div>
          <div class="value hero primary" id="tp-rate">--</div>
        </div>
        <div class="stat-stack">
          <div class="stat">
            <div class="label">False activations</div>
            <div class="value sec" id="fp-count">0</div>
          </div>
          <div class="stat">
            <div class="label">Total detections</div>
            <div class="value sec" id="total-count">0</div>
          </div>
        </div>
      </div>
    </div>
    <div class="panel eval-panel latency-panel">
      <div class="panel-head"><div class="panel-title">Latency</div></div>
      <div class="eval-body">
        <div class="stat">
          <div class="label">Wake &rarr; stream</div>
          <div class="value hero" id="last-latency">-- <span class="unit">ms</span></div>
        </div>
        <div class="stat-stack">
          <div class="stat">
            <div class="label">Average</div>
            <div class="value sec" id="avg-latency">-- <span class="unit">ms</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- 2. LIVE PROCESSING -->
  <div class="panel processing-panel">
    <div class="panel-head">
      <div class="panel-title">Live processing</div>
      <div class="state-badge listening" id="state-badge">LISTENING</div>
    </div>
    <div class="pipeline-body">
      <div class="pipe-cell">
        <div class="cell-label">Live audio waveform</div>
        <canvas id="wave-canvas" width="1000" height="220"></canvas>
      </div>
      <div class="pipe-cell">
        <div class="cell-label">Wake-word spectrogram</div>
        <div class="spec-body">
          <img id="wake-spec" alt="Wake-word window spectrogram" style="display:none">
          <div class="spec-placeholder" id="wake-spec-placeholder">no wake-word window captured yet</div>
        </div>
      </div>
      <div class="pipe-cell">
        <div class="cell-label">Command spectrogram</div>
        <div class="spec-body">
          <img id="cmd-spec" alt="Command audio spectrogram" style="display:none">
          <div class="spec-placeholder" id="cmd-spec-placeholder">waiting for command audio</div>
        </div>
      </div>
    </div>
  </div>

  <!-- 3. LATEST DETECTION + LIVE DETECTIONS -->
  <div class="bottom">
    <div class="panel latest-card">
      <div class="panel-head" style="margin-bottom:0"><div class="panel-title">Latest detection</div></div>
      <div class="ce-top">
        <div class="ce-status-line" id="ce-status-line"><div class="dot"></div><span id="ce-status-text">NO DETECTION YET</span></div>
        <div class="ce-keyword" id="ce-keyword">--</div>
      </div>
      <div class="ce-metrics">
        <div class="ce-metric">
          <div class="label">Confidence</div>
          <div class="value" id="ce-confidence">--</div>
        </div>
        <div class="ce-metric">
          <div class="label">Latency</div>
          <div class="value" id="ce-latency">--</div>
        </div>
      </div>
      <div class="ce-transcript-col">
        <div class="label">Latest transcript</div>
        <div class="text" id="ce-transcript">Waiting for first detection...</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head"><div class="panel-title">Live detections</div></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>RAM</th>
              <th>CPU</th>
              <th>Budget</th>
              <th>Latency</th>
              <th>Conf.</th>
              <th>Transcript</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody id="log-body">
            <tr><td colspan="8" class="empty">Waiting for first detection...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
const WAKE_THRESHOLD_PERCENT = {{ wake_threshold_percent }};
const WAKE_KEYWORD = "ARISE";

const STATE_LABELS = {
  listening: 'LISTENING',
  wake_detected: 'WAKE DETECTED',
  streaming: 'STREAMING',
  transcribing: 'TRANSCRIBING'
};

function formatUptime(sec) {
  if (sec == null) return '--';
  sec = Math.floor(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) {
    return String(h) + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
  }
  return String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
}

function budgetBadge(ok) {
  if (ok) return '<span class="badge ok">&check; WITHIN BUDGET</span>';
  return '<span class="badge bad">&times; OVER BUDGET</span>';
}

function resultTag(row) {
  if (row.status === 'true_positive') return '<span class="tag tp">&check; TRUE POSITIVE</span>';
  if (row.status === 'false_activation') return '<span class="tag fp">&times; FALSE ACTIVATION</span>';
  return '<span class="tag pending">&mdash; PENDING</span> ' +
    '<button class="mark correct" onclick="mark(' + row.id + ',\\'true_positive\\')">&check; CORRECT</button>' +
    '<button class="mark wrong" onclick="mark(' + row.id + ',\\'false_activation\\')">&times; FALSE</button>';
}

function confidenceClass(conf) {
  if (conf == null) return '';
  if (conf < WAKE_THRESHOLD_PERCENT) return 'conf-bad';
  if (conf < WAKE_THRESHOLD_PERCENT + 5) return 'conf-warn';
  return 'conf-ok';
}

async function refresh() {
  try {
    const res = await fetch('/api/data');
    const data = await res.json();

    document.getElementById('conn-dot').className = 'dot' + (data.connected ? ' on' : '');
    document.getElementById('conn-text').textContent = data.connected ? 'CONNECTED' : 'DEVICE DISCONNECTED';
    document.getElementById('status-meta').innerHTML = data.connected
      ? 'ESP32<br>Session ' + formatUptime(data.uptime_sec)
      : '';

    const state = data.pipeline_state || 'listening';
    const stateBadge = document.getElementById('state-badge');
    stateBadge.textContent = STATE_LABELS[state] || state.toUpperCase();
    stateBadge.className = 'state-badge ' + state;

    if (data.latest) {
      const ramPct = Math.min(100, (data.latest.ram_kb / {{ ram_budget }}) * 100);
      const cpuPct = Math.min(100, (data.latest.cpu_percent / {{ cpu_budget }}) * 100);
      const wakeMs = data.latest.stream_start_latency_ms != null ? data.latest.stream_start_latency_ms : data.latest.latency_ms;
      const conf = data.latest.confidence;

      document.getElementById('ram-value').innerHTML = '<span class="metric-num">' + data.latest.ram_kb + '</span><span class="unit">/ {{ ram_budget }} KB</span>';
      document.getElementById('ram-bar').style.width = ramPct + '%';
      document.getElementById('ram-bar').className = 'bar-fill' + (ramPct > 90 ? ' warn' : '');

      document.getElementById('cpu-value').innerHTML = '<span class="metric-num">' + data.latest.cpu_percent + '</span><span class="unit">/ {{ cpu_budget }}%</span>';
      document.getElementById('cpu-bar').style.width = cpuPct + '%';
      document.getElementById('cpu-bar').className = 'bar-fill' + (cpuPct > 90 ? ' warn' : '');

      document.getElementById('last-latency').innerHTML = wakeMs + ' <span class="unit">ms</span>';

      const ceLine = document.getElementById('ce-status-line');
      ceLine.classList.add('active');
      document.getElementById('ce-status-text').textContent = 'WAKE DETECTED';
      document.getElementById('ce-keyword').textContent = WAKE_KEYWORD;
      document.getElementById('ce-confidence').textContent = conf != null ? conf.toFixed(1) + '%' : '--';
      document.getElementById('ce-confidence').className = 'value ' + confidenceClass(conf);
      document.getElementById('ce-latency').textContent = wakeMs + ' ms';
      document.getElementById('ce-transcript').textContent = data.latest.transcript || 'Could not recognize speech.';

      const wakeImg = document.getElementById('wake-spec');
      const wakePh = document.getElementById('wake-spec-placeholder');
      if (data.latest.wake_spectrogram_png) {
        wakeImg.src = data.latest.wake_spectrogram_png;
        wakeImg.style.display = 'block';
        wakePh.style.display = 'none';
      } else {
        wakeImg.removeAttribute('src');
        wakeImg.style.display = 'none';
        wakePh.style.display = 'flex';
      }

      const cmdImg = document.getElementById('cmd-spec');
      const cmdPh = document.getElementById('cmd-spec-placeholder');
      if (data.latest.command_spectrogram_png) {
        cmdImg.src = data.latest.command_spectrogram_png;
        cmdImg.style.display = 'block';
        cmdPh.style.display = 'none';
      } else {
        cmdImg.removeAttribute('src');
        cmdImg.style.display = 'none';
        cmdPh.style.display = 'flex';
      }
    }

    const allRows = data.log || [];

    document.getElementById('total-count').textContent = data.total;
    document.getElementById('fp-count').textContent = data.false_positives;
    document.getElementById('tp-rate').innerHTML = data.tp_rate === null ? '--' : data.tp_rate + '<span class="unit">%</span>';
    document.getElementById('avg-latency').innerHTML = data.avg_latency === null ? '-- <span class="unit">ms</span>' : data.avg_latency + ' <span class="unit">ms</span>';

    const tbody = document.getElementById('log-body');
    if (allRows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="empty">Waiting for first detection...</td></tr>';
    } else {
      tbody.innerHTML = allRows.map(function(row) {
        const wake = row.stream_start_latency_ms != null ? row.stream_start_latency_ms : row.latency_ms;
        const conf = row.confidence != null ? Number(row.confidence).toFixed(1) + '%' : '--';
        const confCls = confidenceClass(row.confidence);

        return '<tr>' +
          '<td>' + row.time + '</td>' +
          '<td>' + row.ram_kb + ' KB</td>' +
          '<td>' + row.cpu_percent + '%</td>' +
          '<td>' + budgetBadge(!!row.within_budget) + '</td>' +
          '<td>' + wake + ' ms</td>' +
          '<td class="' + confCls + '">' + conf + '</td>' +
          '<td class="transcript-cell">' + (row.transcript ? row.transcript.slice(0, 40) : '--') + '</td>' +
          '<td>' + resultTag(row) + '</td>' +
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

    ctx.fillStyle = '#0a0e15';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1c2434';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = data.streaming ? 3 : 2;
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
        model_size=MODEL_SIZE_KB,
        event_name=EVENT_NAME,
        team_name=TEAM_NAME,
        team_id=TEAM_ID,
        ps_id=PS_ID,
        wake_threshold_percent=round(WAKE_THRESHOLD * 100, 2)
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
    log_rows = []
    for d in reversed(detections[-20:]):
        row = dict(d)
        row["wake_spectrogram_png"] = None
        row["command_spectrogram_png"] = None
        log_rows.append(row)

    connected = device_is_connected() or stream_active
    update_disconnect_count(connected)

    uptime_sec = None
    if first_connected_ts is not None:
        uptime_sec = round(time.time() - first_connected_ts, 1)

    return jsonify({
        "connected": connected,
        "uptime_sec": uptime_sec,
        "disconnect_count": disconnect_count,
        "wake_threshold_percent": round(WAKE_THRESHOLD * 100, 2),
        "pipeline_state": get_pipeline_state(),
        "total": total,
        "false_positives": false_positives,
        "tp_rate": tp_rate,
        "avg_latency": avg_latency,
        "latest": latest,
        "log": log_rows,
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
    confidence = request.headers.get("X-Wake-Confidence", None)

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

    command_pcm = None
    if len(request.data) > 44:
        command_pcm = np.frombuffer(request.data[44:], dtype="<i2")

    log_detection(
        ram_kb,
        cpu_percent,
        latency_ms,
        utterance_duration_ms,
        latest_text,
        confidence,
        command_pcm
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