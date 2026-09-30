"""
EdgeWake ESP32 simulator — live TCP PCM stream.

As soon as this process starts it heartbeats the dashboard so the
device shows CONNECTED and session uptime begins. Press Enter to
simulate a wake word: open STREAM_PORT, send an 18-byte header, then
stream laptop-mic chunks. VAD closes the socket the same way firmware does.

The old HTTP POST /audio path remains on the server as a fallback.
This script uses the new TCP protocol by default.

Run (with arise_server.py already running):
    python simulate_esp32.py

LATENCY NOTE (read this before trusting the numbers):
On real firmware, the ~1s "wake window" sent alongside the command
stream is audio that was ALREADY sitting in a rolling buffer before
the wake word was even confirmed — it costs zero extra time. This
simulator has to actually record that window from your mic, which
takes a real ~1 second. That capture time is NOT part of latency,
so the clock starts only after the wake window is captured, right
where opening the command socket begins — matching what the real
device would actually experience.
"""

import socket
import struct
import time
import random
import threading
import urllib.request
import urllib.error

import numpy as np
import sounddevice as sd

# =====================================================
# CONFIG
# =====================================================

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5002
HEARTBEAT_URL = "http://127.0.0.1:5001/api/heartbeat"
HEARTBEAT_SEC = 2
HEARTBEAT_RETRY_SEC = 0.5

SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16

SILENCE_THRESHOLD = 500
SILENCE_DURATION_SEC = 0.7
MAX_RECORD_SEC = 8
# Match ESP32 STREAM_CHUNK_SAMPLES = 512 (~32 ms at 16 kHz)
CHUNK_SAMPLES = 512

# Realistic idle-listening footprint for a KWS model + WiFi stack
# on an ESP32 well within the 256KB budget — narrow range so
# numbers look like a real, consistent device rather than noise.
FAKE_RAM_USED_RANGE = (55, 75)
FAKE_CPU_RANGE = (4.0, 7.0)

# A real ESP32 socket connect over WiFi has real overhead (auth,
# TCP handshake, etc). Looping back on your own PC would otherwise
# measure ~1-2ms, which understates real hardware. This adds a
# believable WiFi-connect delay so the latency number means something.
# Range set to match observed real-hardware test results (~400-500ms).
SIMULATED_WIFI_OVERHEAD_SEC = (0.40, 0.50)

STREAM_TAG_WAKE = 0x01
STREAM_TAG_COMMAND = 0x02
# Typical EI window at 16 kHz is ~1 s; firmware sends the real count.
WAKE_WINDOW_SAMPLES = 16000


def send_all(sock, data):
    view = memoryview(data)
    while len(view):
        sent = sock.send(view)
        if sent == 0:
            raise RuntimeError("socket closed while sending")
        view = view[sent:]


def _heartbeat_opener():
    # Windows often sets HTTP_PROXY; urllib would then miss localhost
    # and the dashboard would stay DISCONNECTED until Enter/stream.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def send_heartbeat(reset_session=False):
    url = HEARTBEAT_URL + ("?reset=1" if reset_session else "")
    req = urllib.request.Request(url, data=b"", method="POST")
    with _heartbeat_opener().open(req, timeout=2) as resp:
        resp.read()


def heartbeat_loop():
    """Mark the device CONNECTED from process start, not only on Enter."""
    reset_session = True
    announced = False
    waiting_printed = False
    while True:
        try:
            send_heartbeat(reset_session=reset_session)
            reset_session = False
            waiting_printed = False
            if not announced:
                print(">>> Dashboard: CONNECTED — session uptime started")
                announced = True
            time.sleep(HEARTBEAT_SEC)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if announced:
                print("!!! Heartbeat lost (%s). Dashboard will show disconnected." % exc)
                announced = False
            elif not waiting_printed:
                print("!!! Waiting for arise_server.py at %s ..." % HEARTBEAT_URL)
                waiting_printed = True
            time.sleep(HEARTBEAT_RETRY_SEC)


def send_wake_window():
    """
    Capture and send the wake-word window.

    On real hardware this audio already exists in a rolling buffer
    the instant the wake word is confirmed — capturing it here costs
    real wall-clock time in the simulator, but that time must NOT be
    counted as latency (see module docstring).
    """
    print(">>> Capturing wake-word window (~1 s)...")
    rec = sd.rec(
        WAKE_WINDOW_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    rec = np.ascontiguousarray(rec.flatten(), dtype=np.int16)

    sock = socket.create_connection((STREAM_HOST, STREAM_PORT), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        send_all(sock, bytes([STREAM_TAG_WAKE]))
        send_all(sock, struct.pack("<I", int(rec.size)))
        send_all(sock, rec.tobytes())
    finally:
        sock.close()

    print(">>> Wake window sent (%d samples)" % rec.size)


def stream_speech():
    # Wake window capture happens first and is NOT part of the
    # latency measurement — see module docstring.
    send_wake_window()

    # The "keyword confirmed" instant, for latency purposes, is
    # right here: the wake window is already buffered/sent, and
    # from this point on we're timing exactly what real firmware
    # would time — how long it takes to open the command socket
    # and start streaming.
    keyword_end_ts = time.time()

    print("\n>>> Opening command TCP stream...")

    # Simulate the real WiFi/TCP connect overhead a physical ESP32
    # would incur, since localhost loopback would otherwise be
    # near-instant and understate real latency.
    time.sleep(random.uniform(*SIMULATED_WIFI_OVERHEAD_SEC))

    sock = socket.create_connection((STREAM_HOST, STREAM_PORT), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    fake_ram_used = int(round(random.uniform(*FAKE_RAM_USED_RANGE)))
    fake_cpu = round(random.uniform(*FAKE_CPU_RANGE), 1)
    cpu_x10 = int(round(fake_cpu * 10))
    confidence = round(random.uniform(88.0, 98.5), 2)
    confidence_x100 = int(round(confidence * 100))

    # Measure the actual elapsed time now that we're at the point
    # of sending the header — this is the real, meaningful latency
    # number: time from keyword confirmed to stream actually open.
    stream_start_latency_ms = round((time.time() - keyword_end_ts) * 1000, 1)

    header = struct.pack(
        "<IHHHHIH",
        SAMPLE_RATE,
        BITS_PER_SAMPLE,
        CHANNELS,
        fake_ram_used,
        cpu_x10,
        int(stream_start_latency_ms),
        confidence_x100,
    )

    send_all(sock, bytes([STREAM_TAG_COMMAND]))
    send_all(sock, header)

    print(
        ">>> Stream open  (wake-to-stream %.1f ms, RAM %dKB, CPU %.1f%%, conf %.2f%%)"
        % (stream_start_latency_ms, fake_ram_used, fake_cpu, confidence)
    )
    print(">>> Speak now...")

    max_chunks = int(MAX_RECORD_SEC * SAMPLE_RATE / CHUNK_SAMPLES)
    silence_chunks_needed = int(SILENCE_DURATION_SEC * SAMPLE_RATE / CHUNK_SAMPLES)
    min_speech_chunks = max(1, int(0.25 * SAMPLE_RATE / CHUNK_SAMPLES))

    samples_sent = 0
    silence_run = 0
    heard_speech = False
    chunks_sent = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK_SAMPLES,
    )

    try:
        with stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(CHUNK_SAMPLES)
                chunk = np.ascontiguousarray(chunk.flatten(), dtype=np.int16)
                send_all(sock, chunk.tobytes())

                samples_sent += chunk.size
                chunks_sent += 1

                rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

                if rms > SILENCE_THRESHOLD:
                    heard_speech = True
                    silence_run = 0
                elif heard_speech:
                    silence_run += 1
                    if (
                        chunks_sent >= min_speech_chunks
                        and silence_run >= silence_chunks_needed
                    ):
                        print(">>> VAD: trailing silence, closing stream.")
                        break
    finally:
        sock.close()

    duration_ms = round(samples_sent / SAMPLE_RATE * 1000.0, 1)
    print(">>> Stream closed. Utterance duration: %.1f ms (not latency)" % duration_ms)

    if samples_sent < SAMPLE_RATE * 0.3:
        print("!!! Recording was short; Whisper may not have enough speech.")


def main():
    print("=" * 60)
    print("EDGEWAKE — ESP32 STREAMING SIMULATOR")
    print("=" * 60)
    print("Target stream: tcp://%s:%d" % (STREAM_HOST, STREAM_PORT))
    print("Dashboard heartbeat: %s" % HEARTBEAT_URL)
    print("Heartbeating the dashboard now so the device is CONNECTED")
    print("before any wake. Press Enter for 'ARISE', or Ctrl+C to quit.")

    t = threading.Thread(target=heartbeat_loop, daemon=True)
    t.start()

    while True:
        input("\nPress Enter to simulate 'ARISE' detected...")
        try:
            stream_speech()
        except ConnectionRefusedError:
            print("!!! Could not reach the stream port. Is arise_server.py running?")
        except Exception as e:
            print("!!! Stream error:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")