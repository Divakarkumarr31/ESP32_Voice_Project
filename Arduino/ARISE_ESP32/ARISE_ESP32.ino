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
 *      - Record 5 seconds of speech
 *      - Send WAV to Mac Flask server
 *      - Restart wake-word detection
 */

#define EIDSP_QUANTIZE_FILTERBANK 0

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

#include <Audio_Classification_-_Keyword_Spotting_inferencing.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s.h"


// =====================================================
// WIFI
// =====================================================

const char* WIFI_SSID = "Dev-2.4g";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

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
// POST-WAKE SPEECH RECORDING
// =====================================================

#define SPEECH_SECONDS 5

#define SPEECH_SAMPLES \
    (SAMPLE_RATE * SPEECH_SECONDS)


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

void connectWiFi();


// =====================================================
// SETUP
// =====================================================

void setup()
{
    Serial.begin(115200);

    delay(2000);


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
}


// =====================================================
// MAIN LOOP
// =====================================================

void loop()
{
    // =================================================
    // GET MICROPHONE DATA
    // =================================================

    bool m =
        microphone_inference_record();


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


    EI_IMPULSE_ERROR r =
        run_classifier_continuous(
            &signal,
            &result,
            debug_nn
        );


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

    if (
        ++print_results >=
        EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW
    )
    {
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


            // =================================================
            // RECORD SPEECH AND SEND
            // =================================================

            bool success =
                recordSpeechAndSend();


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
// RECORD 5 SECONDS OF SPEECH
// AND SEND WAV TO SERVER
// =====================================================

bool recordSpeechAndSend()
{
    Serial.println();

    Serial.println(
        "================================"
    );

    Serial.println(
        "PREPARING TO RECORD SPEECH"
    );

    Serial.println(
        "================================"
    );


    // =================================================
    // STOP EDGE IMPULSE
    // =================================================

    microphone_inference_end();


    delay(300);


    // =================================================
    // ALLOCATE SPEECH BUFFER
    // =================================================

    const int totalSamples =
        SPEECH_SAMPLES;


    int16_t *speechBuffer =
        (int16_t *)malloc(
            totalSamples *
            sizeof(int16_t)
        );


    if (
        speechBuffer == NULL
    )
    {
        Serial.println(
            "ERROR: Could not allocate speech buffer."
        );

        return false;
    }


    // =================================================
    // START I2S
    // =================================================

    if (
        i2s_init(
            SAMPLE_RATE
        ) != 0
    )
    {
        Serial.println(
            "ERROR: Could not start I2S."
        );


        free(
            speechBuffer
        );


        return false;
    }


    i2s_zero_dma_buffer(
        I2S_PORT
    );


    delay(100);


    // =================================================
    // START RECORDING
    // =================================================

    Serial.println();

    Serial.println(
        "================================"
    );

    Serial.println(
        ">>> SPEAK NOW <<<"
    );

    Serial.println(
        "Recording 5 seconds..."
    );

    Serial.println(
        "================================"
    );


    int samplesRecorded = 0;


    int32_t rawBuffer[512];


    size_t bytesRead;


    while (
        samplesRecorded <
        totalSamples
    )
    {
        esp_err_t result =
            i2s_read(
                I2S_PORT,
                rawBuffer,
                sizeof(rawBuffer),
                &bytesRead,
                portMAX_DELAY
            );


        if (
            result != ESP_OK
        )
        {
            Serial.println(
                "ERROR: I2S read failed."
            );


            i2s_deinit();


            free(
                speechBuffer
            );


            return false;
        }


        int samplesRead =
            bytesRead /
            sizeof(int32_t);


        for (
            int i = 0;
            i < samplesRead &&
            samplesRecorded < totalSamples;
            i++
        )
        {
            speechBuffer[
                samplesRecorded++
            ] =
                (int16_t)(
                    rawBuffer[i] >> 16
                );
        }
    }


    // =================================================
    // STOP I2S
    // =================================================

    i2s_deinit();


    Serial.println(
        "Recording complete."
    );


    // =================================================
    // CHECK WIFI
    // =================================================

    if (
        WiFi.status() !=
        WL_CONNECTED
    )
    {
        Serial.println(
            "Wi-Fi disconnected. Reconnecting..."
        );


        connectWiFi();
    }


    if (
        WiFi.status() !=
        WL_CONNECTED
    )
    {
        Serial.println(
            "ERROR: No Wi-Fi connection."
        );


        free(
            speechBuffer
        );


        return false;
    }


    // =================================================
    // WAV INFORMATION
    // =================================================

    const uint32_t dataSize =
        totalSamples *
        sizeof(int16_t);


    const uint32_t wavSize =
        44 +
        dataSize;


    uint8_t wavHeader[44] = {0};


    // RIFF
    memcpy(
        wavHeader,
        "RIFF",
        4
    );


    uint32_t fileSize =
        wavSize - 8;


    memcpy(
        wavHeader + 4,
        &fileSize,
        4
    );


    // WAVE
    memcpy(
        wavHeader + 8,
        "WAVE",
        4
    );


    // fmt
    memcpy(
        wavHeader + 12,
        "fmt ",
        4
    );


    uint32_t fmtSize = 16;


    memcpy(
        wavHeader + 16,
        &fmtSize,
        4
    );


    uint16_t audioFormat = 1;


    memcpy(
        wavHeader + 20,
        &audioFormat,
        2
    );


    uint16_t channels = 1;


    memcpy(
        wavHeader + 22,
        &channels,
        2
    );


    uint32_t sampleRate =
        SAMPLE_RATE;


    memcpy(
        wavHeader + 24,
        &sampleRate,
        4
    );


    uint32_t byteRate =
        SAMPLE_RATE *
        channels *
        sizeof(int16_t);


    memcpy(
        wavHeader + 28,
        &byteRate,
        4
    );


    uint16_t blockAlign =
        channels *
        sizeof(int16_t);


    memcpy(
        wavHeader + 32,
        &blockAlign,
        2
    );


    uint16_t bitsPerSample = 16;


    memcpy(
        wavHeader + 34,
        &bitsPerSample,
        2
    );


    // data
    memcpy(
        wavHeader + 36,
        "data",
        4
    );


    memcpy(
        wavHeader + 40,
        &dataSize,
        4
    );


    // =================================================
    // CONNECT TO FLASK SERVER
    // =================================================

    Serial.println();

    Serial.println(
        "Connecting to Flask server..."
    );


    WiFiClient client;


    if (
        !client.connect(
            "192.168.1.44",
            5001
        )
    )
    {
        Serial.println(
            "ERROR: Could not connect to server."
        );


        free(
            speechBuffer
        );


        return false;
    }


    // =================================================
    // SEND HTTP HEADER
    // =================================================

    client.print(
        "POST /audio HTTP/1.1\r\n"
    );


    client.print(
        "Host: 192.168.1.44:5001\r\n"
    );


    client.print(
        "Content-Type: audio/wav\r\n"
    );


    client.print(
        "Content-Length: "
    );


    client.print(
        wavSize
    );


    client.print(
        "\r\n"
    );


    client.print(
        "Connection: close\r\n"
    );


    client.print(
        "\r\n"
    );


    // =================================================
    // SEND WAV HEADER
    // =================================================

    size_t written =
        client.write(
            wavHeader,
            44
        );


    if (
        written != 44
    )
    {
        Serial.println(
            "ERROR: WAV header send failed."
        );


        client.stop();


        free(
            speechBuffer
        );


        return false;
    }


    // =================================================
    // SEND AUDIO DATA
    // =================================================

    const uint8_t *audioBytes =
        (const uint8_t *)speechBuffer;


    size_t remaining =
        dataSize;


    while (
        remaining > 0
    )
    {
        size_t chunk =
            remaining > 1024
            ? 1024
            : remaining;


        size_t sent =
            client.write(
                audioBytes,
                chunk
            );


        if (
            sent == 0
        )
        {
            Serial.println(
                "ERROR: Audio send failed."
            );


            client.stop();


            free(
                speechBuffer
            );


            return false;
        }


        audioBytes += sent;


        remaining -= sent;
    }


    Serial.println(
        "Audio uploaded."
    );


    // =================================================
    // WAIT FOR SERVER RESPONSE
    // =================================================

    unsigned long timeout =
        millis() + 5000;


    while (
        !client.available()
        &&
        millis() < timeout
    )
    {
        delay(10);
    }


    if (
        client.available()
    )
    {
        String response =
            client.readString();


        Serial.println();

        Serial.println(
            "Server response:"
        );


        Serial.println(
            response
        );
    }
    else
    {
        Serial.println(
            "No response from server."
        );
    }


    client.stop();


    // =================================================
    // FREE BUFFER
    // =================================================

    free(
        speechBuffer
    );


    return true;
}


// =====================================================
// SENSOR CHECK
// =====================================================

#if !defined(EI_CLASSIFIER_SENSOR) || \
    EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE

#error "Invalid model for current sensor."

#endif