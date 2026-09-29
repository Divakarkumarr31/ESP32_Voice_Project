# ARISE - ESP32 Voice Activation System

ARISE is a low-latency voice activation system built using an ESP32, an INMP441 digital MEMS microphone, and an Edge Impulse TinyML keyword-spotting model.

The ESP32 continuously listens for the wake word **ARISE locally**. Wake-word detection runs directly on the ESP32 and does not continuously send microphone audio to the server.

When ARISE is detected with a confidence of **0.80 or higher**, the ESP32:

1. Turns on the onboard LED for approximately 2 seconds.
2. Records 5 seconds of speech.
3. Converts the recording to 16 kHz, 16-bit, mono WAV audio.
4. Sends the WAV audio over Wi-Fi to the Python server.
5. The server performs speech-to-text using Faster Whisper.
6. The recognized text is displayed on a web presentation page.
7. The ESP32 returns to continuous ARISE detection.

---

# System Architecture

```text
                    INMP441
                  MEMS Microphone
                        |
                        | I2S
                        v
                     ESP32
                        |
                        v
               ARISE TinyML Model
                  INT8 / Local
                        |
                        |
                 ARISE >= 0.80
                        |
                        v
                Onboard LED ON
                  ~2 seconds
                        |
                        v
                Record 5 seconds
                   of speech
                        |
                        | Wi-Fi
                        v
                 Python Server
                    Flask
                        |
                        v
                Faster Whisper
                 Speech-to-Text
                        |
                        v
                 Recognized Text
                        |
                        v
              Presentation Web Page
                        |
                        v
              Return to ARISE Detection
```

---

# Hardware Requirements

- ESP32 DOIT DEVKIT V1
- INMP441 digital MEMS microphone
- USB data cable
- Computer for running the Python server
- Wi-Fi network

---

# INMP441 Wiring

| INMP441 Pin | ESP32 |
|---|---|
| VDD | 3.3V |
| GND | GND |
| SCK / BCLK | GPIO 26 |
| WS / LRCL | GPIO 25 |
| SD / DOUT | GPIO 33 |
| L/R | GND |

The INMP441 is configured for the **LEFT** audio channel because L/R is connected to GND.

The onboard LED used for wake-word indication is connected to:

```text
GPIO 2
```

---

# Audio Configuration

The microphone is read using I2S with the following configuration:

```text
Sample rate:        16,000 Hz
I2S sample width:  32-bit
Channel:            ONLY_LEFT

BCLK:               GPIO 26
WS:                 GPIO 25
DOUT:               GPIO 33
```

The INMP441 provides 32-bit I2S samples.

The ESP32 converts these samples to 16-bit PCM before passing them to the Edge Impulse model.

The post-wake speech recording sent to the server is:

```text
Sample rate: 16,000 Hz
Bit depth:   16-bit
Channels:    Mono
Duration:    5 seconds
Format:      WAV
```

---

# Machine Learning Model

The wake-word model was trained and exported using Edge Impulse.

```text
Wake word:          ARISE
Model type:         Keyword Spotting
Quantization:       INT8
Sample rate:        16 kHz
Detection threshold: 0.80

Classes:
- Arise
- noise
- unknown
```

The exported Edge Impulse Arduino library is included in the repository:

```text
dependencies/edge-impulse/
```

The firmware uses the generated Edge Impulse header:

```cpp
#include <Audio_Classification_-_Keyword_Spotting_inferencing.h>
```

The exported model is already compiled for use with the Arduino ESP32 environment.

---

# Repository Structure

```text
ESP32_Voice_Project/
|
|-- Arduino/
|   `-- ARISE_ESP32/
|       `-- ARISE_ESP32.ino
|
|-- dependencies/
|   `-- edge-impulse/
|       `-- ei-audio-classification--keyword-spotting-arduino-1.0.92-impulse-#4.zip
|
|-- recordings/
|   |-- Arise/
|   `-- noise/
|
|-- server/
|   |-- arise_server.py
|   |-- record_arise.py
|   `-- requirements.txt
|
|-- docs/
|
|-- scripts/
|   `-- setup.sh
|
|-- .gitignore
`-- README.md
```

---

# Software Requirements

The project requires:

- Arduino IDE
- ESP32 board support for Arduino IDE
- Python 3
- Git

The Python server uses:

```text
Flask
faster-whisper
```

These are installed automatically from:

```text
server/requirements.txt
```

The ESP32 firmware also requires the exported Edge Impulse Arduino library included in:

```text
dependencies/edge-impulse/
```

---

# 1. Clone the Repository

Clone the repository:

```bash
git clone <REPOSITORY_URL>
```

Enter the project directory:

```bash
cd ESP32_Voice_Project
```

Replace `<REPOSITORY_URL>` with the actual GitHub repository URL.

---

# 2. Python Server Setup

The project includes an automatic setup script:

```text
scripts/setup.sh
```

The script automatically:

1. Enters the `server` directory.
2. Creates a Python virtual environment.
3. Activates the virtual environment during setup.
4. Upgrades pip.
5. Installs the dependencies from `server/requirements.txt`.

First make the script executable:

```bash
chmod +x scripts/setup.sh
```

Then run:

```bash
./scripts/setup.sh
```

After the setup finishes, the Python environment will be located at:

```text
server/.venv/
```

The virtual environment is intentionally excluded from Git using `.gitignore`.

---

# 3. Start the Python Server

Enter the server directory:

```bash
cd server
```

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

Start the server:

```bash
python arise_server.py
```

The Flask server listens on:

```text
Port: 5001
```

The audio endpoint is:

```text
/audio
```

Therefore, the complete endpoint is:

```text
http://SERVER_IP:5001/audio
```

For example:

```text
http://192.168.1.44:5001/audio
```

The IP address must be the local IP address of the computer running the Python server.

The ESP32 and computer must be connected to the same local network.

---

# 4. Python Server Web Interface

The server also provides a presentation web page at:

```text
http://SERVER_IP:5001
```

After the ESP32 sends audio and Whisper recognizes the speech, the recognized text is displayed on this page.

The page automatically refreshes every 2 seconds.

The interface displays:

```text
ARISE

Voice Recognition System

[ Recognized Text ]

ESP32 -> Edge AI -> Wi-Fi -> Whisper
```

---

# 5. Install ESP32 Board Support

Open Arduino IDE.

Go to:

```text
Tools -> Board -> Boards Manager
```

Search for:

```text
ESP32
```

Install:

```text
esp32 by Espressif Systems
```

This provides the ESP32 Arduino framework and built-in ESP32 libraries such as:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include "driver/i2s.h"
```

These libraries do not need to be copied into the project repository.

---

# 6. Install the Edge Impulse Arduino Library

The exported Edge Impulse library ZIP is included in:

```text
dependencies/edge-impulse/
```

The ZIP file is:

```text
ei-audio-classification--keyword-spotting-arduino-1.0.92-impulse-#4.zip
```

In Arduino IDE select:

```text
Sketch -> Include Library -> Add .ZIP Library...
```

Select the ZIP file from:

```text
dependencies/edge-impulse/
```

After installation, Arduino should recognize:

```cpp
#include <Audio_Classification_-_Keyword_Spotting_inferencing.h>
```

The Edge Impulse ZIP is included in the repository so another developer can install the exact exported model used by the project.

---

# 7. Open the ESP32 Firmware

Open:

```text
Arduino/ARISE_ESP32/ARISE_ESP32.ino
```

This is the firmware that runs on the ESP32.

---

# 8. Configure Wi-Fi

Inside the Arduino firmware, configure:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
```

Replace the placeholders with the Wi-Fi credentials used by the ESP32.

The ESP32 and computer running the Python server must be connected to the same network.

Do not commit real Wi-Fi passwords to GitHub.

---

# 9. Configure the Python Server IP in the ESP32

The ESP32 sends the recorded WAV file to the computer running the Flask server.

The server computer's local IP address must be configured in the ESP32 firmware.

For example:

```text
192.168.1.44
```

The ESP32 sends audio to:

```text
http://SERVER_IP:5001/audio
```

Example:

```text
http://192.168.1.44:5001/audio
```

If the computer's IP address changes, update the server IP in the Arduino firmware.

The IP address shown in the Python server's startup message should also be updated if necessary.

---

# 10. Select the ESP32 Board

In Arduino IDE select:

```text
Tools -> Board -> ESP32 Arduino -> DOIT ESP32 DEVKIT V1
```

Then select the connected ESP32 serial port from:

```text
Tools -> Port
```

---

# 11. Upload the ESP32 Firmware

Open:

```text
Arduino/ARISE_ESP32/ARISE_ESP32.ino
```

Click:

```text
Verify
```

If compilation succeeds, click:

```text
Upload
```

After uploading, open the Serial Monitor at:

```text
115200 baud
```

---

# 12. Complete System Startup

The recommended startup order is:

### Step 1

Connect the INMP441 microphone to the ESP32.

### Step 2

Connect the ESP32 to the computer using USB.

### Step 3

Start the Python server:

```bash
cd server
source .venv/bin/activate
python arise_server.py
```

### Step 4

Open Arduino Serial Monitor at:

```text
115200 baud
```

### Step 5

Reset or power the ESP32.

### Step 6

Wait for the ESP32 to connect to Wi-Fi and begin continuous inference.

### Step 7

Speak:

```text
ARISE
```

### Step 8

When ARISE confidence reaches at least `0.80`, the ESP32:

```text
Turns LED ON
       |
       v
Waits approximately 2 seconds
       |
       v
Records 5 seconds of speech
       |
       v
Creates 16-bit mono WAV
       |
       v
Sends WAV over Wi-Fi
```

### Step 9

The Python server receives the WAV and passes it to Faster Whisper.

### Step 10

The recognized text is displayed in the server terminal and on:

```text
http://SERVER_IP:5001
```

---

# Runtime Flow

```text
                         START
                           |
                           v
                     ESP32 Boot
                           |
                           v
                    Connect to Wi-Fi
                           |
                           v
                    Initialize INMP441
                           |
                           v
                  Load Edge Impulse
                     INT8 Model
                           |
                           v
               Continuous ARISE Detection
                           |
              +------------+------------+
              |                         |
              v                         v
       Arise < 0.80               Arise >= 0.80
              |                         |
              |                         v
              |                  Wake Detected
              |                         |
              |                         v
              |                    LED ON
              |                    ~2 sec
              |                         |
              |                         v
              |                  Record 5 sec
              |                         |
              |                         v
              |                   Create WAV
              |                         |
              |                         v
              |                  Send via Wi-Fi
              |                         |
              |                         v
              |                  Flask Server
              |                         |
              |                         v
              |                 Faster Whisper
              |                         |
              |                         v
              |                  Recognized Text
              |                         |
              |                         v
              |                Web Presentation Page
              |                         |
              +-------------------------+
                           |
                           v
                 Return to ARISE Detection
```

---

# Expected Serial Output

A successful wake-word detection should look approximately like:

```text
Predictions:
    Arise: 0.95
    noise: 0.02
    unknown: 0.03

ARISE DETECTED!

Confidence: 0.95

LED ON
LED OFF

================================
PREPARING TO RECORD SPEECH
================================

>>> SPEAK NOW <<<
Recording 5 seconds...

Recording complete.
Audio uploaded.

Server response:
...
```

The exact confidence values will vary between recordings.

---

# Expected Python Server Output

When the ESP32 sends audio, the server should display something similar to:

```text
============================================================
AUDIO RECEIVED FROM ESP32
============================================================

Audio saved: latest_audio.wav
Audio size: 160044 bytes

Running Whisper speech recognition...

============================================================
RECOGNIZED TEXT:
============================================================

Hello this is a test of the ARISE voice assistant

Detected language: en
Language probability: 0.98
```

The exact text and language probability will depend on the recorded speech.

---

# Edge Impulse Prediction

The ESP32 continuously evaluates the incoming microphone audio using the Edge Impulse model.

Typical prediction output:

```text
Predictions:
    Arise: 0.95
    noise: 0.02
    unknown: 0.03
```

The wake word is triggered when:

```text
Arise >= 0.80
```

If the confidence is below the threshold, the ESP32 continues listening without sending the audio to the server.

---

# Dataset

Training recordings are stored in:

```text
recordings/Arise/
recordings/noise/
```

These recordings are used as the project dataset and can be used for future Edge Impulse model training or improvement.

The dataset is **not required for normal runtime** because the trained Edge Impulse model is already exported and included in the project.

---

# Recording Utility

The project also contains:

```text
server/record_arise.py
```

This script is a recording utility used during development and dataset collection.

It is not required for normal wake-word detection or server operation.

The main runtime server is:

```text
server/arise_server.py
```

---

# Python Dependencies

The Python server imports:

```python
from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
```

Therefore `server/requirements.txt` contains:

```text
Flask
faster-whisper
```

Python modules such as:

```text
os
time
```

are part of Python's standard library and do not need to be installed separately.

---

# Troubleshooting

## ESP32 Does Not Upload

Check:

- `DOIT ESP32 DEVKIT V1` is selected.
- Correct serial port is selected.
- USB cable supports data transfer.
- Arduino Serial Monitor is closed during upload.

---

## Edge Impulse Header Is Not Found

If Arduino reports an error such as:

```text
Audio_Classification_-_Keyword_Spotting_inferencing.h:
No such file or directory
```

install the exported Edge Impulse ZIP again:

```text
dependencies/edge-impulse/
```

using:

```text
Sketch -> Include Library -> Add .ZIP Library...
```

---

## ESP32 Does Not Detect ARISE

Check:

```text
BCLK = GPIO 26
WS   = GPIO 25
DOUT = GPIO 33
L/R  = GND
```

Also verify:

```text
Sample rate = 16 kHz
I2S = 32-bit
Channel = ONLY_LEFT
```

Check the Serial Monitor prediction values.

If `Arise` remains near zero while `noise` or `unknown` is high, verify that the microphone configuration matches the configuration used during model testing.

---

## ESP32 Connects to Wi-Fi but Cannot Reach the Server

Check:

- ESP32 and computer are connected to the same Wi-Fi network.
- `arise_server.py` is running.
- Server port is `5001`.
- Server IP configured in the ESP32 is correct.
- Computer firewall is not blocking port `5001`.

The ESP32 must be able to reach:

```text
http://SERVER_IP:5001/audio
```

---

## Server Does Not Start

Make sure the virtual environment is activated:

```bash
cd server
source .venv/bin/activate
```

Then run:

```bash
python arise_server.py
```

If dependencies are missing, run:

```bash
pip install -r requirements.txt
```

---

## Faster Whisper Takes Time to Start

The first time the server starts, Faster Whisper may need to download the selected Whisper model:

```text
base
```

This requires an internet connection during the initial model download.

After the model is available locally, the server can use it without continuously downloading the model.

The server runs Whisper using:

```text
device: CPU
compute type: INT8
```

---

## Server Receives Audio but Speech Is Not Recognized

Check:

- The microphone is recording correctly.
- The 5-second recording contains speech.
- The WAV file is being received by the server.
- The server has successfully loaded Faster Whisper.
- The recorded speech is loud and clear enough.

The server automatically detects the spoken language because no fixed language is specified in the Whisper configuration.

---

# Quick Start

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd ESP32_Voice_Project
```

Make the setup script executable:

```bash
chmod +x scripts/setup.sh
```

Run the setup script:

```bash
./scripts/setup.sh
```

Start the Python server:

```bash
cd server
source .venv/bin/activate
python arise_server.py
```

Then install the Edge Impulse Arduino library from:

```text
dependencies/edge-impulse/
```

using Arduino IDE:

```text
Sketch -> Include Library -> Add .ZIP Library...
```

Open:

```text
Arduino/ARISE_ESP32/ARISE_ESP32.ino
```

Configure:

```text
Wi-Fi SSID
Wi-Fi password
Server IP
```

Select:

```text
DOIT ESP32 DEVKIT V1
```

Upload the firmware.

Open the Serial Monitor at:

```text
115200 baud
```

Speak:

```text
ARISE
```

The complete system should then operate as:

```text
ARISE
  |
  v
ESP32 detects locally
  |
  v
LED indication
  |
  v
5-second speech recording
  |
  v
Wi-Fi upload
  |
  v
Flask
  |
  v
Faster Whisper
  |
  v
Recognized text
  |
  v
Web presentation page
```

---

# Security

Never commit sensitive information to GitHub.

Do not commit:

```text
Wi-Fi passwords
API keys
Private credentials
.env files
Personal tokens
```

Use placeholder values such as:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
```

The Python virtual environment is also not committed:

```text
server/.venv/
```

because it is recreated automatically using:

```text
scripts/setup.sh
```

---

# .gitignore

The project uses `.gitignore` to exclude generated and private files.

The important ignored files/directories include:

```text
server/.venv/
__pycache__/
*.pyc
.DS_Store
.env
.env.*
*.tmp
```

The following project files should remain tracked:

```text
Arduino/
dependencies/
recordings/
server/
scripts/
docs/
README.md
```

---

# Project Status

The current ARISE system includes:

- ESP32 DOIT DEVKIT V1 firmware
- INMP441 digital MEMS microphone
- Local ARISE wake-word detection
- Edge Impulse TinyML model
- INT8 quantized inference
- 16 kHz audio processing
- 32-bit I2S microphone input
- 16-bit PCM conversion
- GPIO 2 LED wake indication
- 0.80 ARISE detection threshold
- Wi-Fi communication
- 5-second post-wake speech capture
- 16-bit mono WAV generation
- Flask server
- Faster Whisper speech-to-text
- Automatic language detection
- Web-based recognized-text presentation
- Python virtual-environment setup script
- Reproducible Python dependency installation
- Edge Impulse model library included in the repository

---

# Project Flow Summary

```text
             ARISE VOICE ACTIVATION SYSTEM

                    INMP441
                       |
                       v
                     ESP32
                       |
                       v
              Edge Impulse INT8
                 Wake Detection
                       |
                ARISE >= 0.80
                       |
                       v
                  LED ON
                       |
                       v
                Record 5 sec
                       |
                       v
                 WAV Audio
                       |
                     Wi-Fi
                       |
                       v
                Flask Server
                       |
                       v
               Faster Whisper
                       |
                       v
                Speech-to-Text
                       |
                       v
               Recognized Text
                       |
                       v
             Web Presentation Page
                       |
                       v
             ARISE Detection Again
```

---