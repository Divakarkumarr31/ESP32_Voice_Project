"""
EdgeWake ESP32 simulator.

Simulates the ESP32 device without any hardware, so you can test
the Flask server + dashboard end-to-end using your laptop mic.

Each time you press Enter, it pretends the wake word was just
detected: it records your speech (stopping automatically after
silence, same VAD idea as the real firmware), builds a WAV file,
attaches fake-but-plausible RAM/CPU/latency headers, and POSTs it
to /audio exactly like the ESP32 does.

Install dependencies first:
    pip install sounddevice numpy requests

Run (with arise_server.py already running):
    python simulate_esp32.py
"""

import io
import time
import wave
import random

import numpy as np
import sounddevice as sd
import requests

# =====================================================
# CONFIG
# =====================================================

SERVER_URL = "http://127.0.0.1:5001/audio"

SAMPLE_RATE = 16000
CHANNELS = 1

SILENCE_THRESHOLD = 500       # same idea as the ESP32 RMS threshold
SILENCE_DURATION_SEC = 0.7    # stop after this much trailing silence
MAX_RECORD_SEC = 8            # hard safety cap, same as firmware
CHUNK_SEC = 0.05              # 50ms chunks for VAD checking

# Fake telemetry ranges — tweak these to whatever your real
# firmware has been logging, so the dashboard looks representative
RAM_BUDGET_KB = 256
FAKE_RAM_USED_RANGE = (150, 210)      # KB used, out of 256
CPU_BUDGET_PERCENT = 10
FAKE_CPU_RANGE = (3.0, 8.5)           # % idle CPU


def record_with_vad():
    """Record from the mic until trailing silence or max duration."""
    print("\n>>> Recording... speak your command now.")

    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)
    max_chunks = int(MAX_RECORD_SEC / CHUNK_SEC)
    silence_chunks_needed = int(SILENCE_DURATION_SEC / CHUNK_SEC)

    recorded = []
    silence_run = 0
    heard_speech = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    with stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            recorded.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

            if rms > SILENCE_THRESHOLD:
                heard_speech = True
                silence_run = 0
            elif heard_speech:
                silence_run += 1
                if silence_run >= silence_chunks_needed:
                    break

    audio = np.concatenate(recorded) if recorded else np.array([], dtype=np.int16)
    duration = len(audio) / SAMPLE_RATE
    print(f">>> Stopped recording ({duration:.2f}s captured).")
    return audio


def audio_to_wav_bytes(audio):
    """Pack int16 PCM samples into an in-memory WAV file."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def send_to_server(wav_bytes, keyword_end_ts):
    fake_ram_used = round(random.uniform(*FAKE_RAM_USED_RANGE), 1)
    fake_cpu = round(random.uniform(*FAKE_CPU_RANGE), 1)
    latency_ms = round((time.time() - keyword_end_ts) * 1000, 1)

    headers = {
        "Content-Type": "audio/wav",
        "X-Device-RAM-KB": str(fake_ram_used),
        "X-Device-CPU-Percent": str(fake_cpu),
        "X-Detection-Latency-MS": str(latency_ms),
    }

    print(f">>> Sending to server  (RAM {fake_ram_used}KB, CPU {fake_cpu}%, latency {latency_ms}ms)")

    try:
        response = requests.post(
            SERVER_URL,
            data=wav_bytes,
            headers=headers,
            timeout=60,
        )
        print(">>> Server response:", response.status_code, response.json())
    except requests.exceptions.ConnectionError:
        print("!!! Could not reach the server. Is arise_server.py running on port 5001?")
    except Exception as e:
        print("!!! Error sending to server:", e)


def main():
    print("=" * 60)
    print("EDGEWAKE — ESP32 SIMULATOR (no hardware needed)")
    print("=" * 60)
    print(f"Target server: {SERVER_URL}")
    print("Press Enter to simulate a wake-word detection, or Ctrl+C to quit.")

    while True:
        input("\nPress Enter to simulate 'ARISE' detected...")

        # This is the moment the real ESP32 would have just
        # confirmed the wake word — latency is measured from here.
        keyword_end_ts = time.time()

        audio = record_with_vad()

        if len(audio) < SAMPLE_RATE * 0.3:
            print("!!! Recording too short, skipping send (probably no speech detected).")
            continue

        wav_bytes = audio_to_wav_bytes(audio)
        send_to_server(wav_bytes, keyword_end_ts)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
