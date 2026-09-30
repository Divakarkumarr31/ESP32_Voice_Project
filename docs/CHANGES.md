# Changes since `main`

This document describes everything added on top of the original ARISE repo
(`main` commit: _Initial ARISE ESP32 voice activation system_).

Use it as a delta against the original README, which still describes the
**record 5 seconds → HTTP WAV POST** baseline.

---

## Baseline (`main`)

The original system:

1. ESP32 listens locally with an Edge Impulse INT8 keyword-spotting model.
2. If class `Arise` ≥ **0.80**, GPIO 2 LED is on for ~2 seconds.
3. Firmware **mallocs a large buffer**, records a **fixed 5 seconds**, builds a WAV, POSTs it to Flask `POST /audio` on port **5001**.
4. Faster Whisper transcribes; a simple auto-refresh HTML page shows the text.

There was no VAD, no device telemetry, no live dashboard, and no TCP streaming.

---

## What we changed (in order)

### 1. VAD recording and device telemetry (firmware)

**Files:** `Arduino/ARISE_ESP32/ARISE_ESP32.ino`

- Replaced the fixed 5-second capture with **RMS-based VAD**:
  - `SILENCE_THRESHOLD` = 500 (tunable)
  - trailing silence ≈ **700 ms**
  - ignore leading silence until speech is heard (`MIN_SPEECH_MS` = 250)
  - hard cap **8 seconds**
- WAV size used the **actual sample count** (when the HTTP path still existed).
- **RAM:** `logFreeHeap()` using `ESP.getFreeHeap()`, printed as  
  `Free heap: X bytes (Y KB used of 256KB budget)`  
  (`Y` ≈ heap-at-boot − current free).
- **CPU:** approximate idle % from capture + classifier time vs `loop()` wall time (not a FreeRTOS idle-task stat).
- **Latency:** `keywordEndMs` at `Arise >= 0.80`. Printed separately from utterance length after streaming was added.

### 2. EdgeWake dashboard (server)

**Files:** `server/arise_server.py`

Replaced the single presentation page with a polling dashboard (`GET /` + `GET /api/data` every ~1.5 s):

- Connection status
- RAM / idle CPU vs budget (`RAM_BUDGET_KB` = 256, `CPU_BUDGET_PERCENT` = 10)
- True-positive / false-activation marking (`POST /api/mark/<id>/<status>`)
- Detection log, last wake-to-stream latency, utterance duration
- Live transcript

Whisper still uses Faster Whisper `base` on CPU INT8.

**WAV decode workaround:** PyAV 19 rejects `av.open(..., metadata_errors=...)`. The server loads 16-bit WAV (or raw PCM) with Python/`numpy` and passes float32 samples to `transcribe()`.

### 3. Laptop ESP32 simulator

**File:** `server/simulate_esp32.py`  
**Deps:** `numpy`, `sounddevice`, `requests` (and later `pillow`)

Lets you exercise the server **without hardware**. Current mode is the **TCP stream** (see below), not the old “record then POST WAV” client.

### 4. Live TCP command streaming

**Firmware + `arise_server.py` + simulator**

Replaced “buffer the whole utterance then upload one WAV” with:

1. Open a raw TCP socket to **`STREAM_PORT` 5002**.
2. Stream int16 PCM chunks as they are captured (VAD / 8 s cap unchanged).
3. Close the socket (**TCP FIN** = end of utterance).

The **250 KB `malloc` speech buffer was removed**. Capture uses a ~512-sample scratch buffer only.

Flask still serves **`POST /audio`** as an HTTP fallback. Do not use it if you want a live waveform.

**Latency split (do not mix these):**

| Field                     | Meaning                                                  |
| ------------------------- | -------------------------------------------------------- |
| `stream_start_latency_ms` | Keyword-end → TCP command stream open (“wake-to-stream”) |
| `utterance_duration_ms`   | How long the user talked (not system latency)            |

**Note:** firmware still waits **~2 s for the LED** before opening the command stream, so hardware wake-to-stream includes that delay. The simulator has no LED.

### 5. Live waveform canvas

- Server keeps a thread-safe rolling RMS buffer filled **inside the TCP `recv` loop**.
- Dashboard polls `GET /api/waveform` every **~120 ms** (separate from `/api/data`).
- HTTP `/audio` does **not** drive a live waveform (only a dump after the POST, if anything).

The simulator process must print `tcp://127.0.0.1:5002`, not `:5001/audio`.

### 6. Confidence, trends, budget, uptime

**18-byte command header** (little-endian), after a type tag (see protocol):

| Offset | Field                                                               |
| ------ | ------------------------------------------------------------------- |
| 0      | `uint32` sample rate (16000)                                        |
| 4      | `uint16` bits (16)                                                  |
| 6      | `uint16` channels (1)                                               |
| 8      | `uint16` ram_kb                                                     |
| 10     | `uint16` cpu × 10                                                   |
| 12     | `uint32` stream_start_latency_ms                                    |
| 16     | `uint16` confidence_x100 (`ariseConfidence * 10000`, range 0–10000) |

Server stores confidence as a **percent** (e.g. 96.42). Dashboard:

- Confidence column (green if ≥ `WAKE_THRESHOLD` × 100 = **80%**, amber if within 5 points of threshold)
- **Trend** canvas from last ≤20 detections (confidence / RAM% / CPU%) — updated in `refresh()`, **not** the 120 ms waveform loop
- **Budget** badge: `ram_kb ≤ 256` and `cpu_percent ≤ 10`
- **Uptime:** `first_connected_ts` (set once), `disconnect_count` only on connected → disconnected

`WAKE_THRESHOLD = 0.80` in Python must stay aligned with firmware.

### 7. Wake-word window + two mel spectrograms

**Firmware**

- Independent ring buffer, **not** Edge Impulse’s `inference.buffers`:
  - `WAKE_WINDOW_SAMPLES = EI_CLASSIFIER_SLICE_SIZE * EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW`
  - Written in `audio_inference_callback` alongside EI samples
- On wake, **snapshot the ring into chronological order before the LED delay** (so 2 s of LED does not overwrite the window)
- `microphone_inference_end()` still runs later, inside `streamSpeechToServer()`

The ring + snapshot are **static BSS** (~`4 * WAKE_WINDOW_SAMPLES` bytes), not heap. At ~16 k samples that is ~64 KB extra RAM.

**Protocol on port 5002**

| Tag    | Meaning                                                             |
| ------ | ------------------------------------------------------------------- |
| `0x01` | Wake window: then `uint32` sample count + raw int16 PCM, then close |
| `0x02` | Command stream: then 18-byte header + PCM until FIN                 |

Wake audio is stored as `latest_wake_window` and attached to the **next** command detection.

**Dashboard — Feature Extraction**

- “Wake-Word Window (what triggered detection)”
- “Command Audio (what's being transcribed)”

Log-mel images are generated with numpy FFT + a mel filterbank + **Pillow** (no librosa). If the wake message is missing, the UI shows a placeholder.

The simulator sends a ~1 s mic capture as tag `0x01`, then the live command stream as `0x02`.

### 8. Dashboard redesign and team header

**File:** `server/arise_server.py` (only the dashboard HTML/CSS/JS changed)

- Three-tier card hierarchy (title / primary value / context), one spacing scale, tabular numerals.
- Fonts: **JetBrains Mono + Inter** from Google Fonts, with system-font fallbacks so it still works offline.
- System Resources: RAM, CPU and Model Size as three equal metrics (the "within budget" badge was removed from that panel; the per-row badge in the table stays).
- Bottom row: **Latest Detection** card (left, 37%) + **Live Detections** table (right, 63%). The transcript is the most prominent element of the card.
- Table: **Duration column removed**; "Wake→stream" column relabelled **Latency**.
- Navbar: larger `EDGEWAKE` logo plus a team strip (event, team name, team ID, problem statement). Values live in the config block at the top of `arise_server.py`:

```python
EVENT_NAME = "SIH 2026"
TEAM_NAME = "Neural Nomads1"
TEAM_ID = "183857"
PS_ID = "26172"
```

### 9. Cloud answer + spoken reply (Ollama + pyttsx3)

**File:** `server/arise_server.py`  
**New dependency:** `pyttsx3` (Python package). **Ollama** is a separate program.

After Whisper finishes and the detection is logged, a **background thread**:

1. Sends the transcript to a **local Ollama** model over HTTP (`http://localhost:11434/api/chat`, using Python's built-in `urllib`, so no `ollama` pip package is needed).
2. Stores the one-sentence answer on the detection (`response`, `response_status`: `off` / `pending` / `done` / `failed`, plus `response_error` on failure).
3. Speaks the answer **on the server** with offline `pyttsx3` (the ESP32 has no speaker, and this keeps the edge RAM/CPU budget untouched).

Design rules:

- Runs **after** the transcript is logged, so it never delays wake, stream, transcription, or the latency numbers.
- Failures stay soft (dashboard shows `No response (<reason>)`; console prints exception type/message).
- Dashboard: a small cyan **Response** line under the transcript in the Latest Detection card ("Thinking…" while pending).
- The model is warmed up in the background when the server starts (3 attempts, 5 s apart).
- urllib uses a no-proxy opener so Windows system proxies cannot hijack `localhost`.
- Startup lists Ollama tags and warns if `OLLAMA_MODEL` is missing.

Switches and settings at the top of `arise_server.py`:

```python
ENABLE_LLM_ANSWER = True      # False = skip the whole answer stage
ENABLE_SPEECH = True          # False = show text only, no audio
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # must match `ollama list`
LLM_TIMEOUT_SEC = 90          # first load on CPU can exceed 20 s
SPEECH_RATE = 165
```

Set `ENABLE_LLM_ANSWER = False` when recording the scored evaluation numbers (idle CPU, latency, true-positive rate).

---

### 10. Ollama answer reliability (proxy / timeout / warm-up)

**File:** `server/arise_server.py`

Fixes for `ask_llm()` returning `None` while Ollama itself was healthy:

- Bypass system HTTP proxies on all Ollama calls (`ProxyHandler({})`).
- Raise `LLM_TIMEOUT_SEC` to **90** (model load + Whisper on CPU).
- Richer console errors (`type(e).__name__`, HTTP body on `HTTPError`).
- Warm-up retries (3×, 5 s apart) with `LLM warm-up OK` / `LLM warm-up failed: …`.
- Dashboard `response_error` (e.g. `timeout`, `connection refused`, `model not found`).
- Startup `GET /api/tags` warns if `OLLAMA_MODEL` is not installed.
- Dashboard **Speak** button (`POST /api/speak/<id>`) re-plays the latest answer via `pyttsx3` on the server speakers.

## Current runtime architecture

```text
INMP441 → ESP32 → Edge Impulse (ARISE ≥ 0.80)
                → snapshot last model window
                → TCP 0x01 wake PCM  → server latest_wake_window
                → LED ~2 s
                → TCP 0x02 + header + live PCM (VAD / 8 s)
                → Flask 5001 dashboard + Whisper on command close
                → (background) Ollama answer → dashboard Response + spoken on server
```

---

## How to run — full command list

Repo root in these examples: `D:\Coding\sih` (change the path if yours is different).

### 0. One-time: Python virtualenv and packages

`server/requirements.txt` must now include `pyttsx3`:

```text
Flask
faster-whisper
numpy
sounddevice
requests
pillow
pyttsx3
av>=11,<16
```

**Windows (PowerShell)**

```powershell
cd D:\Coding\sih\server
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `Activate.ps1` is blocked, the `Set-ExecutionPolicy` line is only for that PowerShell process.

**macOS / Linux**

```bash
cd /path/to/sih
chmod +x scripts/setup.sh
./scripts/setup.sh
cd server
source .venv/bin/activate
```

`scripts/setup.sh` creates `server/.venv` and installs `server/requirements.txt`. After that, always `source .venv/bin/activate` before running Python.

**Linux only:** `pyttsx3` needs the system speech engine:

```bash
sudo apt install espeak-ng
```

Windows and macOS use the built-in voices, so nothing extra is needed.

The first Flask start may **download the Whisper `base` model** (needs internet once).

---

### 0b. One-time: install Ollama and pull the model

1. Download and install **Ollama** from `https://ollama.com` (free, open-source).
2. Pull the model (about 2 GB, internet needed only for this step):

```powershell
ollama pull llama3.2:3b
```

3. Confirm the model is installed:

```powershell
ollama list
```

The name in the list must match `OLLAMA_MODEL` in `arise_server.py`.

---

### 1. Start Ollama, then the Flask server (required)

Keep both running.

**1a. Ollama** (skip if the desktop app is already running in the background on Windows/macOS):

```powershell
ollama serve
```

Check it is up: open `http://localhost:11434` in a browser (it says "Ollama is running"), or:

```powershell
curl http://localhost:11434
```

**1b. Flask server** (a separate terminal from `ollama serve`)

**Windows**

```powershell
cd D:\Coding\sih\server
.\.venv\Scripts\Activate.ps1
python arise_server.py
```

**macOS / Linux**

```bash
cd /path/to/sih/server
source .venv/bin/activate
python arise_server.py
```

You should see something like:

- Dashboard: `http://127.0.0.1:5001`
- Audio HTTP fallback: `http://127.0.0.1:5001/audio`
- Live PCM stream: `tcp://0.0.0.0:5002`

If port 5001 or 5002 is in use, stop the other process or change the ports in `arise_server.py` **and** in the firmware / simulator.

**Windows Firewall:** allow Python inbound on **5001** and **5002** if the ESP32 is on another machine.

---

### 2. Open the dashboard

In a browser:

```text
http://127.0.0.1:5001
```

From a phone on the same Wi-Fi, use this PC’s LAN IP instead of `127.0.0.1` (same port). After code changes, hard-refresh the page (`Ctrl+F5`).

---

### 3. Laptop simulator (no ESP32)

Use a **new** terminal. The server from step 1 must already be running.

**Windows**

```powershell
cd D:\Coding\sih\server
.\.venv\Scripts\Activate.ps1
python simulate_esp32.py
```

**macOS / Linux**

```bash
cd /path/to/sih/server
source .venv/bin/activate
python simulate_esp32.py
```

Confirm the banner says:

```text
Target stream: tcp://127.0.0.1:5002
```

If it still says `http://127.0.0.1:5001/audio`, you are on an old process — Ctrl+C and run `python simulate_esp32.py` again.

Then:

1. Allow **microphone** access if the OS asks.
2. Press **Enter** (simulated wake word).
3. Wait ~1 second (wake-window capture).
4. Speak the command (for example, "What is the capital of France?"); it stops after silence or 8 seconds.
5. Watch the dashboard: waveform, spectrograms, transcript, then the **Response** line ("Thinking…" then the answer), and listen for the spoken reply.

Stop the simulator: **Ctrl+C**.

---

### 4. Real ESP32 firmware

Do this only when you have the board. Server (step 1) should run on the PC **first**.

1. Install **Arduino IDE**, **esp32 by Espressif Systems**, board **DOIT ESP32 DEVKIT V1**.
2. Install the Edge Impulse ZIP:

   `dependencies/edge-impulse/ei-audio-classification--keyword-spotting-arduino-1.0.92-impulse-#4.zip`  
   **Sketch → Include Library → Add .ZIP Library...**

3. Open `Arduino/ARISE_ESP32/ARISE_ESP32.ino`.
4. Set Wi-Fi and the **PC’s current LAN IP** (not `127.0.0.1` — the ESP32 cannot reach that):

```cpp
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_HOST = "192.168.x.x";   // PC running arise_server.py
#define STREAM_PORT 5002
```

On Windows, find the PC IP:

```powershell
ipconfig
```

Use the **IPv4 Address** of the Wi-Fi adapter (same network as the ESP32).

5. **Tools → Port** → the ESP32 COM port. Serial Monitor: **115200**.
6. Click **Verify**, then **Upload**. Close Serial Monitor during upload if the port is busy.
7. Power the ESP32, wait for Wi-Fi, say **ARISE**, then the command after the LED.

PC and ESP32 must be on the **same LAN**. The ESP32 talks to `SERVER_HOST:5002` (PCM) and optionally `:5001` (HTTP fallback). Ollama and speech run only on the PC; the ESP32 never talks to them.

**Speaker placement:** keep the PC speaker volume moderate and away from the INMP441 mic. A spoken reply that re-triggers the wake word would count as a false activation.

---

### 5. Recommended order every session

```text
1. ollama serve            (or confirm the Ollama app is running)
2. python arise_server.py
3. Browser → http://127.0.0.1:5001
4. Either python simulate_esp32.py
   or power/upload the ESP32
```

Start Ollama and the server a few minutes before a demo so the model is already loaded.

After pulling new code, stop both Python processes, reinstall deps if `requirements.txt` changed, start the server, then the simulator.

```powershell
cd D:\Coding\sih\server
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python arise_server.py
```

---

### 6. Optional checks

Health of the data API (server must be up):

```powershell
curl http://127.0.0.1:5001/api/data
```

```powershell
curl http://127.0.0.1:5001/api/waveform
```

Ollama is running and the model exists:

```powershell
curl http://localhost:11434
ollama list
```

Quick manual test of the model, outside the project:

```powershell
ollama run llama3.2:3b "What is the capital of France? Answer in one sentence."
```

Stop Flask / simulator / Ollama: **Ctrl+C** in that terminal.

---

### 7. Typical failures

| Symptom                                                        | What to run / check                                                                                           |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Activate.ps1` cannot be loaded                                | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate again                              |
| `ModuleNotFoundError: PIL`                                     | `pip install pillow` or `pip install -r requirements.txt`                                                     |
| `ModuleNotFoundError: pyttsx3` / console prints `Speech error` | `pip install pyttsx3`; on Linux also `sudo apt install espeak-ng`                                             |
| Simulator `ConnectionRefusedError`                             | `python arise_server.py` is not running, or port 5002 blocked                                                 |
| Waveform never moves                                           | Simulator must be TCP (`:5002`); restart both processes; hard-refresh the browser                             |
| ESP32 cannot reach server                                      | Wrong `SERVER_HOST`, different Wi-Fi, or firewall on 5002                                                     |
| Whisper `metadata_errors`                                      | Use current `arise_server.py` (PCM/WAV loaded without PyAV `open`)                                            |
| Response says "No response (…)"                                | Read the reason in parentheses / console `LLM error:`; check Ollama, model name, proxy, or timeout            |
| First answer is very slow                                      | Model still loading; wait for warm-up (`LLM warm-up OK`) or ask again; `LLM_TIMEOUT_SEC` is 90                 |
| Answer shows but no sound                                      | `pip install pyttsx3` in the venv, restart server; check console for `Speech error:` / `Speaking:`            |
| No Response line activity at all                               | `ENABLE_LLM_ANSWER = False` in `arise_server.py`, or the transcript was "Could not recognize speech."         |
| Speak button stays disabled                                    | Wait until Response status is `done` (answer text visible); `ENABLE_SPEECH` must be `True`                     |

---

## Config cheat sheet

| Item                  | Value                                         |
| --------------------- | --------------------------------------------- |
| Flask                 | `0.0.0.0:5001`                                |
| PCM stream            | `5002`                                        |
| Ollama                | `http://localhost:11434`, model `llama3.2:3b` |
| Wake threshold        | 0.80                                          |
| VAD silence           | 700 ms, RMS 500                               |
| Max command           | 8 s                                           |
| RAM / CPU budgets     | 256 KB, 10% idle CPU (as displayed)           |
| LLM / speech switches | `ENABLE_LLM_ANSWER`, `ENABLE_SPEECH`          |

---

## Git branches (as of this write-up)

| Branch                             | Contents                                                                                                       |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `main`                             | Original 5 s WAV + simple Flask page                                                                           |
| `feature/edgewake-vad-dashboard`   | VAD, metrics, dashboard, HTTP simulator era                                                                    |
| `feature/edgewake-live-tcp-stream` | TCP streaming + live waveform (push further commits for spectrograms / 18-byte header if they are still local) |

Uncommitted local work after the last push may include confidence/trend/budget/uptime, wake-window spectrograms, the dashboard redesign, and the Ollama answer stage — keep this file in the same commit as those sources.

---

## Files touched vs `main`

- `Arduino/ARISE_ESP32/ARISE_ESP32.ino`
- `server/arise_server.py`
- `server/simulate_esp32.py` (new)
- `server/requirements.txt` — Flask, faster-whisper, numpy, sounddevice, requests, pillow, **pyttsx3**, `av>=11,<16`
- `.gitignore` — `server/latest_audio.wav`, `*.wav`

---

## Intentionally unchanged

- Wake-word model and Edge Impulse inference path (aside from the extra ring-buffer write)
- INMP441 pinout and 16 kHz / 32-bit I2S → 16-bit conversion
- True/false-positive mark buttons
- HTTP `/audio` kept as fallback until hardware streaming is fully signed off
- ESP32 firmware for the answer stage: none; Ollama and speech run only on the PC
