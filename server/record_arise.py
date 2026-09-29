import serial
import time
import wave
import os


# =====================================================
# ESP32 SETTINGS
# =====================================================

PORT = "/dev/cu.usbserial-120"

CONTROL_BAUD = 115200
AUDIO_BAUD = 460800


# =====================================================
# AUDIO SETTINGS
# =====================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 10
BYTES_PER_SAMPLE = 2

TOTAL_BYTES = (
    SAMPLE_RATE
    * RECORD_SECONDS
    * BYTES_PER_SAMPLE
)


# =====================================================
# NOISE DATASET
# =====================================================

OUTPUT_FOLDER = "recordings/noise"

START_NUMBER = 1


# =====================================================
# OPEN ESP32
# =====================================================

print()
print("========================================")
print("        NOISE DATASET RECORDER")
print("========================================")
print()

print("Opening ESP32...")

ser = serial.Serial(
    PORT,
    CONTROL_BAUD,
    timeout=2
)

time.sleep(4)

ser.reset_input_buffer()

print("ESP32 connection ready.")
print()


# =====================================================
# CREATE FOLDER
# =====================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =====================================================
# RECORDING LOOP
# =====================================================

number = START_NUMBER


try:

    while True:

        filename = os.path.join(
            OUTPUT_FOLDER,
            f"noise_{number:03d}.wav"
        )

        print()
        print("========================================")
        print(f"Recording: noise_{number:03d}.wav")
        print("========================================")
        print()

        print("Press ENTER to start.")
        input()


        # ---------------------------------------------
        # SEND RECORD COMMAND
        # ---------------------------------------------

        print()
        print("Sending recording command...")

        ser.baudrate = CONTROL_BAUD

        ser.write(b"R")
        ser.flush()


        # ---------------------------------------------
        # WAIT FOR ESP32
        # ---------------------------------------------

        print("Waiting for ESP32...")

        ok = False

        wait_start = time.time()

        while time.time() - wait_start < 3:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if line == "OK":
                ok = True
                break


        if not ok:

            print()
            print("ERROR: ESP32 did not respond with OK.")
            break


        print("ESP32 accepted command.")


        # ---------------------------------------------
        # SWITCH TO AUDIO BAUD
        # ---------------------------------------------

        print("Switching to 460800 baud...")

        ser.baudrate = AUDIO_BAUD

        time.sleep(0.7)


        # ---------------------------------------------
        # START AUDIO
        # ---------------------------------------------

        ser.write(b"S")
        ser.flush()


        # ---------------------------------------------
        # RECEIVE AUDIO
        # ---------------------------------------------

        print()
        print("========================================")
        print(">>> RECORDING NOW <<<")
        print("========================================")
        print()

        print("DO NOT SPEAK.")
        print("Let the environment/noise be recorded.")
        print()


        audio_data = bytearray()

        transfer_start = time.time()

        MAX_TRANSFER_TIME = 20


        while len(audio_data) < TOTAL_BYTES:

            if (
                time.time() - transfer_start
                > MAX_TRANSFER_TIME
            ):

                print()
                print("ERROR: Audio transfer timed out.")

                print(
                    f"Received {len(audio_data)} "
                    f"of {TOTAL_BYTES} bytes."
                )

                raise RuntimeError(
                    "Audio transfer timed out."
                )


            remaining = (
                TOTAL_BYTES
                - len(audio_data)
            )


            chunk = ser.read(
                min(4096, remaining)
            )


            if chunk:
                audio_data.extend(chunk)


        # ---------------------------------------------
        # VERIFY
        # ---------------------------------------------

        print()

        print(
            f"Received {len(audio_data)} "
            f"of {TOTAL_BYTES} bytes."
        )


        if len(audio_data) != TOTAL_BYTES:

            raise RuntimeError(
                "Incorrect audio size."
            )


        # ---------------------------------------------
        # SAVE WAV
        # ---------------------------------------------

        with wave.open(
            filename,
            "wb"
        ) as wav:

            wav.setnchannels(1)

            wav.setsampwidth(2)

            wav.setframerate(SAMPLE_RATE)

            wav.writeframes(audio_data)


        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        print()
        print("========================================")
        print("       NOISE RECORDING SUCCESSFUL")
        print("========================================")
        print()

        print(f"Saved: {filename}")
        print(f"Audio bytes: {len(audio_data)}")
        print(f"Sample rate: {SAMPLE_RATE} Hz")
        print(f"Duration: {RECORD_SECONDS} seconds")
        print()


        # ---------------------------------------------
        # RETURN TO CONTROL MODE
        # ---------------------------------------------

        print("Returning to 115200...")

        ser.baudrate = CONTROL_BAUD

        time.sleep(0.5)


        number += 1


except KeyboardInterrupt:

    print()
    print()
    print("Stopping noise recorder...")


except Exception as e:

    print()
    print(f"ERROR: {e}")


finally:

    ser.close()

    print()
    print("Serial connection closed.")
    print("Noise recorder stopped.")