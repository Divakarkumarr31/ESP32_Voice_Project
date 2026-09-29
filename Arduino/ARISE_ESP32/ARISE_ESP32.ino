/*
 * ESP32 ARISE Wake Word + Voice Capture
 *
 * Hardware:
 * INMP441
 * BCLK -> GPIO 26
 * WS   -> GPIO 25
 * DOUT -> GPIO 33
 * L/R  -> GND (LEFT)
 *
 * Operation:
 *
 * 1. Continuously listen for ARISE
 * 2. If ARISE >= 0.80:
 *      - LED ON for 2 seconds
 *      - Open a TCP stream to the server
 *      - Stream I2S chunks live until trailing silence (VAD), max 8 seconds
 *      - Close the socket (TCP FIN = end of utterance)
 *      - Restart wake-word detection
 */

#define EIDSP_QUANTIZE_FILTERBANK 0

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <math.h>

#include <Audio_Classification_-_Keyword_Spotting_inferencing.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"


// =====================================================
// WIFI
// =====================================================

const char* WIFI_SSID = "Dev-2.4g";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* SERVER_HOST = "192.168.1.44";

#define STREAM_PORT 5002

const char* SERVER_URL =
    "http://192.168.1.44:5001/audio";


// =====================================================
// LED
// =====================================================

#define LED_PIN 2

#define WAKE_THRESHOLD 0.80f
#define LED_TIME_MS 2000


// =====================================================
// INMP441
// =====================================================

#define I2S_PORT I2S_NUM_1

#define I2S_BCLK 26
#define I2S_WS   25
#define I2S_DOUT 33

#define SAMPLE_RATE 16000


// =====================================================
// POST-WAKE SPEECH RECORDING (VAD)
// =====================================================

// Worst-case capture length. Recording can stop earlier
// once trailing silence is detected.
#define MAX_SPEECH_SECONDS 8

#define SPEECH_SAMPLES \
    (SAMPLE_RATE * MAX_SPEECH_SECONDS)

// I2S conversion scratch buffer (int16). A few KB, not 250KB.
#define STREAM_CHUNK_SAMPLES 512

// RMS below this counts as silence (tunable).
#define SILENCE_THRESHOLD 500.0f

// Stop after this much consecutive silence (~600-800 ms).
#define SILENCE_DURATION_MS 700

// Ignore leading silence so VAD does not stop immediately.
#define MIN_SPEECH_MS 250


// =====================================================
// EDGE IMPULSE INFERENCE
// =====================================================

typedef struct
{
    signed short *buffers[2];

    unsigned char buf_select;

    unsigned char buf_ready;

    unsigned int buf_count;

    unsigned int n_samples;

} inference_t;


static inference_t inference;


static const uint32_t sample_buffer_size = 2048;


static signed short sampleBuffer[
    sample_buffer_size
];


static bool debug_nn = false;


static int print_results =
    -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);


static bool record_status = true;


// Heap free at boot. Used RAM is approximated as
// (heap-at-boot - current free heap).
static uint32_t heapAtBoot = 0;

// Last computed idle-CPU approximation for HTTP headers.
// Not a FreeRTOS idle-task / scheduler statistic.
static float lastIdleCpuPercent = 0.0f;

static uint32_t lastFreeHeapBytes = 0;

static unsigned long keywordEndMs = 0;

static float lastAriseConfidence = 0.0f;

#define STREAM_TAG_WAKE 0x01
#define STREAM_TAG_COMMAND 0x02

#define WAKE_WINDOW_SAMPLES \
    (EI_CLASSIFIER_SLICE_SIZE * EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)

static int16_t wakeWindowRing[WAKE_WINDOW_SAMPLES];
static uint32_t wakeWindowWritePos = 0;
static int16_t wakeWindowSnapshot[WAKE_WINDOW_SAMPLES];


// =====================================================
// FUNCTION DECLARATIONS
// =====================================================

static void audio_inference_callback(
    uint32_t n_bytes
);

static void capture_samples(
    void* arg
);

static bool microphone_inference_start(
    uint32_t n_samples
);

static bool microphone_inference_record();

static int microphone_audio_signal_get_data(
    size_t offset,
    size_t length,
    float *out_ptr
);

static void microphone_inference_end();

static int i2s_init(
    uint32_t sampling_rate
);

static int i2s_deinit();

bool recordSpeechAndSend();

bool streamSpeechToServer();

bool sendWakeWindowToServer();

void snapshotWakeWindow();

static bool writeAll(
    WiFiClient &client,
    const uint8_t *data,
    size_t length
);

void connectWiFi();

static uint32_t logFreeHeap();


static uint32_t logFreeHeap()
{
    uint32_t freeHeap = ESP.getFreeHeap();

    lastFreeHeapBytes = freeHeap;

    uint32_t used = 0;

    if (heapAtBoot > freeHeap)
    {
        used = heapAtBoot - freeHeap;
    }

    uint32_t usedKb = used / 1024;

    Serial.print("Free heap: ");
    Serial.print(freeHeap);
    Serial.print(" bytes (");
    Serial.print(usedKb);
    Serial.println(" KB used of 256KB budget)");

    return freeHeap;
}


// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);

    heapAtBoot = ESP.getFreeHeap();

    lastFreeHeapBytes = heapAtBoot;


    // =================================================
    // LED
    // =================================================

    pinMode(
        LED_PIN,
        OUTPUT
    );

    digitalWrite(
        LED_PIN,
        LOW
    );


    // =================================================
    // START MESSAGE
    // =================================================

    Serial.println();

    Serial.println(
        "================================"
    );

    Serial.println(
        "ESP32 ARISE Wake Word Detection"
    );

    Serial.println(
        "================================"
    );


    // =================================================
    // WIFI
    // =================================================

    connectWiFi();


    // =================================================
    // EDGE IMPULSE INFORMATION
    // =================================================

    ei_printf(
        "Inferencing settings:\n"
    );

    ei_printf(
        "\tInterval: "
    );

    ei_printf_float(
        (float)EI_CLASSIFIER_INTERVAL_MS
    );

    ei_printf(
        " ms.\n"
    );

    ei_printf(
        "\tFrame size: %d\n",
        EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE
    );

    ei_printf(
        "\tSample length: %d ms.\n",
        EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16
    );

    ei_printf(
        "\tNo. of classes: %d\n",
        sizeof(
            ei_classifier_inferencing_categories
        ) /
        sizeof(
            ei_classifier_inferencing_categories[0]
        )
    );


    // =================================================
    // INITIALIZE CLASSIFIER
    // =================================================

    run_classifier_init();


    ei_printf(
        "\nStarting continuous inference in 2 seconds...\n"
    );

    ei_sleep(2000);


    // =================================================
    // START MICROPHONE INFERENCE
    // =================================================

    if (
        microphone_inference_start(
            EI_CLASSIFIER_SLICE_SIZE
        ) == false
    )
    {
        ei_printf(
            "ERR: Could not allocate audio buffer\n"
        );

        return;
    }


    ei_printf(
        "Recording...\n"
    );

    ei_printf(
        "Say ARISE to test the model.\n"
    );

    ei_printf(
        "Wake window samples: %d (%.0f ms)\n",
        WAKE_WINDOW_SAMPLES,
        (WAKE_WINDOW_SAMPLES * 1000.0f) / SAMPLE_RATE
    );
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
    unsigned long loopStartUs = micros();


    // =================================================
    // GET MICROPHONE DATA
    // =================================================

    unsigned long captureStartUs = micros();

    bool m =
        microphone_inference_record();

    unsigned long captureEndUs = micros();


    if (!m)
    {
        ei_printf(
            "ERR: Failed to record audio...\n"
        );

        return;
    }


    // =================================================
    // CREATE SIGNAL
    // =================================================

    signal_t signal;


    signal.total_length =
        EI_CLASSIFIER_SLICE_SIZE;


    signal.get_data =
        &microphone_audio_signal_get_data;


    // =================================================
    // RUN CLASSIFIER
    // =================================================

    ei_impulse_result_t result = {0};

    const bool logClassifierHeap =
        (print_results + 1 >=
         EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

    if (logClassifierHeap)
    {
        logFreeHeap();
    }

    unsigned long classifierStartUs = micros();

    EI_IMPULSE_ERROR r =
        run_classifier_continuous(
            &signal,
            &result,
            debug_nn
        );

    unsigned long classifierEndUs = micros();

    if (logClassifierHeap)
    {
        logFreeHeap();
    }


    if (
        r != EI_IMPULSE_OK
    )
    {
        ei_printf(
            "ERR: Failed to run classifier (%d)\n",
            r
        );

        return;
    }


    // =================================================
    // PRINT PREDICTIONS
    // =================================================

    bool printedPredictions = false;
    bool didWake = false;

    if (
        ++print_results >=
        EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW
    )
    {
        printedPredictions = true;
        ei_printf(
            "\nPredictions:\n"
        );


        for (
            size_t ix = 0;
            ix < EI_CLASSIFIER_LABEL_COUNT;
            ix++
        )
        {
            ei_printf(
                "    %s: ",
                result.classification[ix].label
            );


            ei_printf_float(
                result.classification[ix].value
            );


            ei_printf(
                "\n"
            );
        }


        ei_printf(
            "    DSP: %d ms\n",
            result.timing.dsp
        );


        ei_printf(
            "    Classification: %d ms\n",
            result.timing.classification
        );


        // =================================================
        // FIND ARISE CONFIDENCE
        // =================================================

        float ariseConfidence = 0.0f;


        for (
            size_t ix = 0;
            ix < EI_CLASSIFIER_LABEL_COUNT;
            ix++
        )
        {
            if (
                strcmp(
                    result.classification[ix].label,
                    "Arise"
                ) == 0
            )
            {
                ariseConfidence =
                    result.classification[ix].value;
            }
        }


        // =================================================
        // WAKE WORD DETECTED
        // =================================================

        if (
            ariseConfidence >=
            WAKE_THRESHOLD
        )
        {
            didWake = true;

            // Keyword-end reference for latency (before LED / VAD / upload).
            keywordEndMs = millis();
            lastAriseConfidence = ariseConfidence;

            snapshotWakeWindow();

            ei_printf(
                "Wake window snapshot: %d samples\n",
                WAKE_WINDOW_SAMPLES
            );

            ei_printf(
                "\n================================\n"
            );

            ei_printf(
                "ARISE DETECTED!\n"
            );

            ei_printf(
                "Confidence: "
            );

            ei_printf_float(
                ariseConfidence
            );

            ei_printf(
                "\n================================\n"
            );


            // =================================================
            // LED ON
            // =================================================

            digitalWrite(
                LED_PIN,
                HIGH
            );


            ei_printf(
                "LED ON\n"
            );


            delay(
                LED_TIME_MS
            );


            digitalWrite(
                LED_PIN,
                LOW
            );


            ei_printf(
                "LED OFF\n"
            );

            sendWakeWindowToServer();

            bool success =
                streamSpeechToServer();


            if (success)
            {
                ei_printf(
                    "\nSpeech sent successfully.\n"
                );
            }
            else
            {
                ei_printf(
                    "\nFailed to send speech.\n"
                );
            }


            // =================================================
            // RESTART WAKE-WORD INFERENCE
            // =================================================

            ei_printf(
                "\nReturning to ARISE detection...\n"
            );


            print_results =
                -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);


            if (
                microphone_inference_start(
                    EI_CLASSIFIER_SLICE_SIZE
                ) == false
            )
            {
                ei_printf(
                    "ERR: Could not restart microphone inference.\n"
                );

                return;
            }
        }


        print_results = 0;
    }


    // Approx idle CPU from this loop iteration.
    // busy = I2S capture-wait in loop() + classifier time.
    // This is wall-clock share, not a true RTOS idle-task statistic.
    if (!didWake)
    {
        unsigned long loopEndUs = micros();

        unsigned long totalLoopUs = loopEndUs - loopStartUs;

        unsigned long busyUs =
            (captureEndUs - captureStartUs) +
            (classifierEndUs - classifierStartUs);

        float cpuBusyPercent = 0.0f;

        if (totalLoopUs > 0)
        {
            cpuBusyPercent =
                (100.0f * (float)busyUs) /
                (float)totalLoopUs;
        }

        if (cpuBusyPercent > 100.0f)
        {
            cpuBusyPercent = 100.0f;
        }

        lastIdleCpuPercent = 100.0f - cpuBusyPercent;

        if (lastIdleCpuPercent < 0.0f)
        {
            lastIdleCpuPercent = 0.0f;
        }

        if (printedPredictions)
        {
            Serial.print("Approx idle CPU: ");
            Serial.print(lastIdleCpuPercent, 1);
            Serial.println("%");
        }
    }
}


// =====================================================
// AUDIO INFERENCE CALLBACK
// =====================================================

static void audio_inference_callback(
    uint32_t n_bytes
)
{
    for (
        int i = 0;
        i < (n_bytes >> 1);
        i++
    )
    {
        inference.buffers[
            inference.buf_select
        ][
            inference.buf_count++
        ] =
            sampleBuffer[i];

        wakeWindowRing[wakeWindowWritePos] = sampleBuffer[i];
        wakeWindowWritePos++;
        if (wakeWindowWritePos >= (uint32_t)WAKE_WINDOW_SAMPLES)
        {
            wakeWindowWritePos = 0;
        }


        if (
            inference.buf_count >=
            inference.n_samples
        )
        {
            inference.buf_select ^= 1;

            inference.buf_count = 0;

            inference.buf_ready = 1;
        }
    }
}


// =====================================================
// CAPTURE TASK
// =====================================================

static void capture_samples(
    void* arg
)
{
    int32_t rawBuffer[512];

    size_t bytes_read = 0;


    while (
        record_status
    )
    {
        i2s_read(
            I2S_PORT,
            rawBuffer,
            sizeof(rawBuffer),
            &bytes_read,
            100
        );


        if (
            bytes_read <= 0
        )
        {
            ei_printf(
                "Error in I2S read\n"
            );

            continue;
        }


        int samples_read =
            bytes_read /
            sizeof(int32_t);


        // Convert 32-bit INMP441 samples
        // to 16-bit PCM
        for (
            int i = 0;
            i < samples_read;
            i++
        )
        {
            sampleBuffer[i] =
                (int16_t)(
                    rawBuffer[i] >> 16
                );
        }


        if (
            record_status
        )
        {
            audio_inference_callback(
                samples_read *
                sizeof(int16_t)
            );
        }
        else
        {
            break;
        }
    }


    vTaskDelete(NULL);
}


// =====================================================
// START MICROPHONE INFERENCE
// =====================================================

static bool microphone_inference_start(
    uint32_t n_samples
)
{
    inference.buffers[0] =
        (signed short *)malloc(
            n_samples *
            sizeof(signed short)
        );


    if (
        inference.buffers[0] == NULL
    )
    {
        return false;
    }


    inference.buffers[1] =
        (signed short *)malloc(
            n_samples *
            sizeof(signed short)
        );


    if (
        inference.buffers[1] == NULL
    )
    {
        ei_free(
            inference.buffers[0]
        );

        return false;
    }


    inference.buf_select = 0;

    inference.buf_count = 0;

    inference.n_samples = n_samples;

    inference.buf_ready = 0;


    if (
        i2s_init(
            EI_CLASSIFIER_FREQUENCY
        )
    )
    {
        ei_printf(
            "Failed to start I2S!\n"
        );


        ei_free(
            inference.buffers[0]
        );

        ei_free(
            inference.buffers[1]
        );


        return false;
    }


    ei_sleep(100);


    record_status = true;


    xTaskCreate(
        capture_samples,
        "CaptureSamples",
        1024 * 32,
        NULL,
        10,
        NULL
    );


    return true;
}


// =====================================================
// WAIT FOR INFERENCE AUDIO
// =====================================================

static bool microphone_inference_record()
{
    if (
        inference.buf_ready == 1
    )
    {
        ei_printf(
            "Error sample buffer overrun.\n"
        );

        return false;
    }


    while (
        inference.buf_ready == 0
    )
    {
        delay(1);
    }


    inference.buf_ready = 0;


    return true;
}


// =====================================================
// GET AUDIO DATA
// =====================================================

static int microphone_audio_signal_get_data(
    size_t offset,
    size_t length,
    float *out_ptr
)
{
    numpy::int16_to_float(
        &inference.buffers[
            inference.buf_select ^ 1
        ][offset],
        out_ptr,
        length
    );


    return 0;
}


// =====================================================
// STOP MICROPHONE INFERENCE
// =====================================================

static void microphone_inference_end()
{
    record_status = false;


    delay(100);


    i2s_deinit();


    ei_free(
        inference.buffers[0]
    );


    ei_free(
        inference.buffers[1]
    );


    inference.buffers[0] = NULL;

    inference.buffers[1] = NULL;

    inference.buf_ready = 0;

    inference.buf_count = 0;
}


// =====================================================
// I2S INITIALIZATION
// =====================================================

static int i2s_init(
    uint32_t sampling_rate
)
{
    i2s_config_t i2s_config = {

        .mode =
            (i2s_mode_t)(
                I2S_MODE_MASTER |
                I2S_MODE_RX
            ),

        .sample_rate =
            sampling_rate,

        // IMPORTANT:
        // INMP441 is read as 32-bit I2S
        .bits_per_sample =
            I2S_BITS_PER_SAMPLE_32BIT,

        .channel_format =
            I2S_CHANNEL_FMT_ONLY_LEFT,

        .communication_format =
            I2S_COMM_FORMAT_I2S,

        .intr_alloc_flags =
            ESP_INTR_FLAG_LEVEL1,

        .dma_buf_count = 8,

        .dma_buf_len = 512,

        .use_apll = false,

        .tx_desc_auto_clear = false,

        .fixed_mclk = 0
    };


    i2s_pin_config_t pin_config = {

        .bck_io_num =
            I2S_BCLK,

        .ws_io_num =
            I2S_WS,

        .data_out_num =
            I2S_PIN_NO_CHANGE,

        .data_in_num =
            I2S_DOUT
    };


    esp_err_t ret;


    ret =
        i2s_driver_install(
            I2S_PORT,
            &i2s_config,
            0,
            NULL
        );


    if (
        ret != ESP_OK
    )
    {
        ei_printf(
            "Error in i2s_driver_install\n"
        );

        return ret;
    }


    ret =
        i2s_set_pin(
            I2S_PORT,
            &pin_config
        );


    if (
        ret != ESP_OK
    )
    {
        ei_printf(
            "Error in i2s_set_pin\n"
        );

        i2s_driver_uninstall(
            I2S_PORT
        );

        return ret;
    }


    i2s_zero_dma_buffer(
        I2S_PORT
    );


    return 0;
}


// =====================================================
// I2S DEINITIALIZATION
// =====================================================

static int i2s_deinit()
{
    i2s_driver_uninstall(
        I2S_PORT
    );


    return 0;
}


// =====================================================
// WIFI CONNECTION
// =====================================================

void connectWiFi()
{
    Serial.println();

    Serial.println(
        "Connecting to Wi-Fi..."
    );


    WiFi.mode(
        WIFI_STA
    );


    WiFi.begin(
        WIFI_SSID,
        WIFI_PASSWORD
    );


    int attempts = 0;


    while (
        WiFi.status() != WL_CONNECTED
        &&
        attempts < 30
    )
    {
        delay(500);

        Serial.print(".");

        attempts++;
    }


    Serial.println();


    if (
        WiFi.status() ==
        WL_CONNECTED
    )
    {
        Serial.println(
            "Wi-Fi connected!"
        );


        Serial.print(
            "ESP32 IP: "
        );


        Serial.println(
            WiFi.localIP()
        );
    }
    else
    {
        Serial.println(
            "Wi-Fi connection FAILED."
        );
    }
}


// =====================================================
// WRITE ALL BYTES ON AN OPEN SOCKET
// =====================================================

static bool writeAll(
    WiFiClient &client,
    const uint8_t *data,
    size_t length
)
{
    size_t remaining = length;

    while (remaining > 0)
    {
        if (!client.connected())
        {
            return false;
        }

        size_t sent = client.write(data, remaining);

        if (sent == 0)
        {
            return false;
        }

        data += sent;
        remaining -= sent;
    }

    return true;
}


void snapshotWakeWindow()
{
    uint32_t pos = wakeWindowWritePos;

    for (uint32_t i = 0; i < (uint32_t)WAKE_WINDOW_SAMPLES; i++)
    {
        wakeWindowSnapshot[i] = wakeWindowRing[pos];
        pos++;
        if (pos >= (uint32_t)WAKE_WINDOW_SAMPLES)
        {
            pos = 0;
        }
    }
}


bool sendWakeWindowToServer()
{
    Serial.println();
    Serial.println("Sending wake-word window...");
    Serial.print("WAKE_WINDOW_SAMPLES=");
    Serial.println(WAKE_WINDOW_SAMPLES);
    logFreeHeap();

    if (WiFi.status() != WL_CONNECTED)
    {
        connectWiFi();
    }

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("ERROR: No Wi-Fi for wake window.");
        return false;
    }

    WiFiClient client;
    client.setNoDelay(true);

    if (!client.connect(SERVER_HOST, STREAM_PORT))
    {
        Serial.println("ERROR: Could not connect for wake window.");
        return false;
    }

    uint8_t tag = STREAM_TAG_WAKE;
    uint32_t count = (uint32_t)WAKE_WINDOW_SAMPLES;

    if (!writeAll(client, &tag, 1))
    {
        client.stop();
        return false;
    }

    if (!writeAll(client, (const uint8_t *)&count, 4))
    {
        client.stop();
        return false;
    }

    if (!writeAll(
            client,
            (const uint8_t *)wakeWindowSnapshot,
            (size_t)WAKE_WINDOW_SAMPLES * sizeof(int16_t)
        ))
    {
        Serial.println("ERROR: Wake window send failed.");
        client.stop();
        return false;
    }

    client.stop();
    Serial.println("Wake window sent.");
    return true;
}
//
// Protocol (little-endian):
//   uint8  tag = 0x02 (command stream)
//   uint32 sample_rate
//   uint16 bits_per_sample
//   uint16 channels
//   uint16 ram_kb
//   uint16 cpu_percent * 10
//   uint32 stream_start_latency_ms
//   uint16 confidence_x100  (ariseConfidence * 10000, 0-10000)
//   then raw int16 PCM until TCP close (FIN)
// =====================================================

bool streamSpeechToServer()
{
    Serial.println();
    Serial.println("================================");
    Serial.println("STREAMING SPEECH TO SERVER");
    Serial.println("================================");

    microphone_inference_end();

    Serial.println("Heap after stopping inference (no 250KB buffer):");
    logFreeHeap();

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("Wi-Fi disconnected. Reconnecting...");
        connectWiFi();
    }

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("ERROR: No Wi-Fi connection.");
        return false;
    }

    WiFiClient client;
    client.setNoDelay(true);

    Serial.println("Opening TCP stream...");

    if (!client.connect(SERVER_HOST, STREAM_PORT))
    {
        Serial.println("ERROR: Could not connect to stream port.");
        return false;
    }

    unsigned long streamStartLatencyMs =
        millis() - keywordEndMs;

    Serial.print("Wake-to-stream latency: ");
    Serial.print(streamStartLatencyMs);
    Serial.println(" ms");

    uint8_t tag = STREAM_TAG_COMMAND;
    if (!writeAll(client, &tag, 1))
    {
        Serial.println("ERROR: Command tag send failed.");
        client.stop();
        return false;
    }

    uint32_t sampleRate = SAMPLE_RATE;
    uint16_t bitsPerSample = 16;
    uint16_t channels = 1;
    uint16_t ramKb = (uint16_t)(lastFreeHeapBytes / 1024);
    uint16_t cpuX10 = (uint16_t)(lastIdleCpuPercent * 10.0f + 0.5f);
    uint32_t latencyMs = (uint32_t)streamStartLatencyMs;
    uint16_t confidenceX100 = (uint16_t)(
        lastAriseConfidence * 10000.0f + 0.5f
    );

    uint8_t header[18];
    memcpy(header + 0, &sampleRate, 4);
    memcpy(header + 4, &bitsPerSample, 2);
    memcpy(header + 6, &channels, 2);
    memcpy(header + 8, &ramKb, 2);
    memcpy(header + 10, &cpuX10, 2);
    memcpy(header + 12, &latencyMs, 4);
    memcpy(header + 16, &confidenceX100, 2);

    Serial.print("Wake confidence x100: ");
    Serial.println(confidenceX100);

    if (!writeAll(client, header, sizeof(header)))
    {
        Serial.println("ERROR: Stream header send failed.");
        client.stop();
        return false;
    }

    if (i2s_init(SAMPLE_RATE) != 0)
    {
        Serial.println("ERROR: Could not start I2S.");
        client.stop();
        return false;
    }

    i2s_zero_dma_buffer(I2S_PORT);

    Serial.println();
    Serial.println("================================");
    Serial.println(">>> SPEAK NOW <<<");
    Serial.println("Streaming until silence (max 8 seconds)...");
    Serial.println("================================");

    const int totalSamples = SPEECH_SAMPLES;
    int samplesRecorded = 0;
    uint32_t consecutiveSilenceSamples = 0;
    bool heardSpeech = false;

    const uint32_t silenceNeedSamples =
        (SAMPLE_RATE * SILENCE_DURATION_MS) / 1000;
    const uint32_t minSpeechSamples =
        (SAMPLE_RATE * MIN_SPEECH_MS) / 1000;

    int32_t rawBuffer[STREAM_CHUNK_SAMPLES];
    int16_t pcmChunk[STREAM_CHUNK_SAMPLES];
    size_t bytesRead;
    uint32_t lastHeapLogSamples = 0;

    while (samplesRecorded < totalSamples)
    {
        esp_err_t result = i2s_read(
            I2S_PORT,
            rawBuffer,
            sizeof(rawBuffer),
            &bytesRead,
            portMAX_DELAY
        );

        if (result != ESP_OK)
        {
            Serial.println("ERROR: I2S read failed.");
            i2s_deinit();
            client.stop();
            return false;
        }

        int samplesRead = bytesRead / sizeof(int32_t);

        if (samplesRead <= 0)
        {
            continue;
        }

        int64_t sumSquares = 0;
        int chunkCount = 0;

        for (
            int i = 0;
            i < samplesRead && samplesRecorded < totalSamples;
            i++
        )
        {
            int16_t sample = (int16_t)(rawBuffer[i] >> 16);
            pcmChunk[chunkCount++] = sample;
            samplesRecorded++;
            sumSquares += (int32_t)sample * (int32_t)sample;
        }

        if (chunkCount <= 0)
        {
            break;
        }

        if (!writeAll(
                client,
                (const uint8_t *)pcmChunk,
                (size_t)chunkCount * sizeof(int16_t)
            ))
        {
            Serial.println("ERROR: Audio stream send failed.");
            i2s_deinit();
            client.stop();
            return false;
        }

        float rms = sqrtf((float)sumSquares / (float)chunkCount);

        if (rms < SILENCE_THRESHOLD)
        {
            consecutiveSilenceSamples += (uint32_t)chunkCount;
        }
        else
        {
            consecutiveSilenceSamples = 0;
            heardSpeech = true;
        }

        if (
            heardSpeech &&
            samplesRecorded >= (int)minSpeechSamples &&
            consecutiveSilenceSamples >= silenceNeedSamples
        )
        {
            Serial.println("VAD: trailing silence reached, stopping.");
            break;
        }

        if (samplesRecorded - (int)lastHeapLogSamples >= SAMPLE_RATE)
        {
            lastHeapLogSamples = (uint32_t)samplesRecorded;
            Serial.println("Heap during speech streaming:");
            logFreeHeap();
        }
    }

    if (samplesRecorded >= totalSamples)
    {
        Serial.println("VAD: 8 second safety cap reached.");
    }

    i2s_deinit();
    client.stop();

    unsigned long utteranceMs =
        ((unsigned long)samplesRecorded * 1000UL) / SAMPLE_RATE;

    Serial.println("Stream closed (TCP FIN).");
    Serial.print("Samples streamed: ");
    Serial.println(samplesRecorded);
    Serial.print("Utterance duration: ");
    Serial.print(utteranceMs);
    Serial.println(" ms");
    Serial.println("Heap after stream (still no 250KB buffer):");
    logFreeHeap();

    return samplesRecorded > 0;
}


bool recordSpeechAndSend()
{
    return streamSpeechToServer();
}

// =====================================================
// SENSOR CHECK
// =====================================================

#if !defined(EI_CLASSIFIER_SENSOR) || \
    EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE

#error "Invalid model for current sensor."

#endif