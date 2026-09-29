"""
EdgeWake ESP32 simulator — live TCP PCM stream.

Opens a socket to STREAM_PORT as soon as you press Enter (wake word),
sends a 16-byte header, then streams laptop-mic chunks while you speak.
VAD closes the socket the same way the firmware does.

The old HTTP POST /audio path remains on the server as a fallback.
This script uses the new TCP protocol by default.

Run (with arise_server.py already running):
    python simulate_esp32.py
"""

import socket
import struct
import time
import random

import numpy as np
import sounddevice as sd

# =====================================================
# CONFIG
# =====================================================

STREAM_HOST = "127.0.0.1"
STREAM_PORT = 5002

SAMPLE_RATE = 16000
CHANNELS = 1
BITS_PER_SAMPLE = 16

SILENCE_THRESHOLD = 500
SILENCE_DURATION_SEC = 0.7
MAX_RECORD_SEC = 8
# Match ESP32 STREAM_CHUNK_SAMPLES = 512 (~32 ms at 16 kHz)
CHUNK_SAMPLES = 512

FAKE_RAM_USED_RANGE = (40, 90)
FAKE_CPU_RANGE = (3.0, 8.5)


def send_all(sock, data):
    view = memoryview(data)
    while len(view):
        sent = sock.send(view)
        if sent == 0:
            raise RuntimeError("socket closed while sending")
        view = view[sent:]


def stream_speech(keyword_end_ts):
    print("\n>>> Opening TCP stream...")

    sock = socket.create_connection((STREAM_HOST, STREAM_PORT), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    stream_start_latency_ms = round((time.time() - keyword_end_ts) * 1000, 1)
    fake_ram_used = int(round(random.uniform(*FAKE_RAM_USED_RANGE)))
    fake_cpu = round(random.uniform(*FAKE_CPU_RANGE), 1)
    cpu_x10 = int(round(fake_cpu * 10))

    header = struct.pack(
        "<IHHHHI",
        SAMPLE_RATE,
        BITS_PER_SAMPLE,
        CHANNELS,
        fake_ram_used,
        cpu_x10,
        int(stream_start_latency_ms),
    )
    send_all(sock, header)

    print(
        ">>> Stream open  (wake-to-stream %.1f ms, RAM %dKB, CPU %.1f%%)"
        % (stream_start_latency_ms, fake_ram_used, fake_cpu)
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
    print("Press Enter to simulate a wake-word detection, or Ctrl+C to quit.")

    while True:
        input("\nPress Enter to simulate 'ARISE' detected...")
        keyword_end_ts = time.time()
        try:
            stream_speech(keyword_end_ts)
        except ConnectionRefusedError:
            print("!!! Could not reach the stream port. Is arise_server.py running?")
        except Exception as e:
            print("!!! Stream error:", e)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
