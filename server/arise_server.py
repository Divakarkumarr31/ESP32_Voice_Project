from flask import Flask, request, jsonify, render_template_string
from faster_whisper import WhisperModel
import os
import time


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
# LATEST RESULT
# =====================================================

latest_text = "Waiting for ARISE..."


# =====================================================
# PRESENTATION WEB PAGE
# =====================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <meta
        http-equiv="refresh"
        content="2"
    >

    <title>ARISE Voice Assistant</title>


    <style>

        body {

            margin: 0;

            background: #111;

            color: white;

            font-family: Arial, Helvetica, sans-serif;

            height: 100vh;

            display: flex;

            align-items: center;

            justify-content: center;

        }


        .container {

            width: 90%;

            max-width: 1400px;

            text-align: center;

        }


        .title {

            font-size: 72px;

            font-weight: bold;

            margin-bottom: 20px;

        }


        .subtitle {

            font-size: 32px;

            color: #aaa;

            margin-bottom: 50px;

        }


        .box {

            background: #222;

            border-radius: 25px;

            padding: 70px;

            min-height: 250px;

            display: flex;

            align-items: center;

            justify-content: center;

            box-shadow: 0 0 30px rgba(255,255,255,0.08);

        }


        .text {

            font-size: 56px;

            line-height: 1.4;

            font-weight: 500;

            word-wrap: break-word;

        }


        .status {

            margin-top: 35px;

            font-size: 26px;

            color: #888;

        }

    </style>

</head>


<body>

    <div class="container">

        <div class="title">
            ARISE
        </div>


        <div class="subtitle">
            Voice Recognition System
        </div>


        <div class="box">

            <div class="text">

                {{ text }}

            </div>

        </div>


        <div class="status">

            ESP32 → Edge AI → Wi-Fi → Whisper

        </div>

    </div>

</body>

</html>
"""


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():

    return render_template_string(
        HTML_PAGE,
        text=latest_text
    )


# =====================================================
# AUDIO ENDPOINT
# =====================================================

@app.route(
    "/audio",
    methods=["POST"]
)
def receive_audio():

    global latest_text


    print()
    print("=" * 60)
    print("AUDIO RECEIVED FROM ESP32")
    print("=" * 60)


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

        segments, info = model.transcribe(

            filename,

            # IMPORTANT:
            # No language is specified.
            # Whisper automatically detects the language.

            task="transcribe",

            beam_size=5,

            best_of=5,

            temperature=0,

            # Ignore silence
            vad_filter=True,

            vad_parameters={

                "min_silence_duration_ms": 500,

                "speech_pad_ms": 200

            },

            # Reduce hallucinations
            no_speech_threshold=0.6,

            log_prob_threshold=-1.0,

            compression_ratio_threshold=2.4

        )


        # =================================================
        # COLLECT TEXT
        # =================================================

        text_parts = []


        for segment in segments:

            text = segment.text.strip()

            if text:

                text_parts.append(text)


        latest_text = " ".join(
            text_parts
        ).strip()


        # =================================================
        # EMPTY RESULT
        # =================================================

        if not latest_text:

            latest_text = "Could not recognize speech."


        # =================================================
        # PRINT RESULT
        # =================================================

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


        # =================================================
        # RETURN RESULT TO ESP32
        # =================================================

        return jsonify({

            "status": "success",

            "text": latest_text,

            "language": info.language

        })


    except Exception as e:

        print(
            "Whisper error:",
            str(e)
        )


        latest_text = (
            "Speech recognition error: "
            + str(e)
        )


        return jsonify({

            "status": "error",

            "text": str(e)

        }), 500


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":

    print()

    print("=" * 60)

    print("ARISE SERVER")

    print("=" * 60)

    print()

    print("Presentation page:")

    print(
        "http://10.75.166.53:5001"
    )

    print()

    print("Audio endpoint:")

    print(
        "http://10.75.166.53:5001/audio"
    )

    print()

    print("Server starting...")

    print()


    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )