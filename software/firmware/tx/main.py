import serial
import time
import base64
import subprocess
import os
import sys
import binascii
import datapacker
import psutil


try:
    from picamera2 import Picamera2, Preview
except ImportError:
    print("Error: picamera2 not found. Please install it ('pip install picamera2')")
    print(
        "or ensure you are running this script on a Raspberry Pi with a camera module."
    )
    sys.exit(1)

try:
    import busio
    from digitalio import DigitalInOut, Direction
    import board
    import adafruit_rfm9x
except ImportError:
    print("Error: Adafruit LoRa libraries not found. Please install them:")
    print("  pip install adafruit-circuitpython-rfm9x adafruit-circuitpython-busdevice")
    print("or ensure you have the necessary hardware (RFM9x module) connected.")
    sys.exit(1)

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE = 9600
READ_TIMEOUT = 1

COMPIG_PATH = "./compig"
INPUT_IMAGE_PATH = "input.jpg"
COMPRESSED_FILE_PATH = "outputs_pi/compressed_image.bin"
OUTPUT_DIR_PI = "outputs_pi"
COMPIG_ENCODE_ARGS = [
    "encode",
    "-input",
    INPUT_IMAGE_PATH,
    "-output",
    COMPRESSED_FILE_PATH,
]

CS = DigitalInOut(board.CE1)
RESET = DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

MAX_LORA_PAYLOAD_BYTES = 250

ser = None
rfm9x = None
picam2 = None


def get_cpu_temp():
    temp = psutil.sensors_temperatures()["cpu_thermal"][0].current
    return temp


def initialize_hardware():
    global ser, rfm9x, picam2

    print(f"Attempting to open serial port: {SERIAL_PORT} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=READ_TIMEOUT,
        )
        print(f"Successfully opened {SERIAL_PORT}.")
    except serial.SerialException as e:
        print(f"Error: Could not open or communicate with serial port {SERIAL_PORT}.")
        print(
            f"Please ensure the port exists and you have sufficient permissions (e.g., add user to 'dialout' group: 'sudo usermod -a -G dialout $USER')."
        )
        print(f"Details: {e}")
        sys.exit(1)

    try:
        rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 910.0)
        rfm9x.tx_power = 23
        rfm9x.spreading_factor = 7
        rfm9x.frequency_mhz = 910.0
        rfm9x.coding_rate = 6
        rfm9x.low_datarate_optimize = True
        rfm9x.enable_crc = True

        print("RFM9x radio initialized successfully.")
    except Exception as e:
        print(f"Error initializing RFM9x radio: {e}")
        print(
            "Please check wiring (SPI, CS, RESET) and ensure 'adafruit-circuitpython-rfm9x' is installed."
        )
        sys.exit(1)

    try:
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(
            main={"size": (568, 320)}, lores={"size": (568, 320)}, display="lores"
        )
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(2)
        print("Picamera2 initialized successfully.")
    except Exception as e:
        print(f"Error initializing Picamera2: {e}")
        print(
            "Please ensure picamera2 is installed and the camera module is enabled (e.g., in raspi-config)."
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR_PI, mode=0o755, exist_ok=True)


def cleanup_hardware():
    global ser, rfm9x, picam2
    print("Performing hardware cleanup...")
    if ser and ser.is_open:
        ser.close()
        print(f"Serial port {SERIAL_PORT} closed.")
    if picam2 and picam2.started:
        picam2.stop()
        print("Picamera2 stopped.")


def capture_and_encode_image():
    if not picam2:
        print("Picamera2 not initialized. Cannot capture image.")
        return None

    try:
        print(f"Capturing image to {INPUT_IMAGE_PATH}...")
        picam2.capture_file(INPUT_IMAGE_PATH)
        print("Image captured.")

        print(
            f"Encoding image with compig: {COMPIG_PATH} {' '.join(COMPIG_ENCODE_ARGS)}"
        )

        if not os.path.exists(COMPIG_PATH):
            print(f"Error: compig executable not found at {COMPIG_PATH}.")
            print(
                f"Please ensure it's compiled and moved to this location or provide its full path."
            )
            return None
        if not os.access(COMPIG_PATH, os.X_OK):
            print(
                f"Error: compig at {COMPIG_PATH} is not executable. Please run 'chmod +x {COMPIG_PATH}'."
            )
            return None

        process = subprocess.run(
            [COMPIG_PATH] + COMPIG_ENCODE_ARGS,
            capture_output=True,
            text=True,
            check=False,
        )

        if process.returncode != 0:
            print(f"compig encoding failed with exit code {process.returncode}")
            print(f"compig stdout: {process.stdout}")
            print(f"compig stderr: {process.stderr}")
            return None

        print("Image encoded by compig.")

        with open(COMPRESSED_FILE_PATH, "rb") as f:
            compressed_data = f.read()

        encoded_string = base64.b64encode(compressed_data).decode("utf-8")
        print(
            f"Compressed data base64 encoded. Size: {len(encoded_string)} characters."
        )
        return encoded_string

    except FileNotFoundError:
        print(
            f"Error: '{COMPIG_PATH}' not found. Make sure 'compig' is in the same directory or provide its full path."
        )
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running compig: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during image processing: {e}")
        return None


def send_lora_message(message_string, message_type="unknown"):
    if not rfm9x:
        print("RFM9x radio not initialized. Cannot send data.")
        return

    if not message_string:
        print(f"No {message_type} to send via LoRa radio.")
        return

    data_to_send_bytes = message_string.encode("utf-8")
    total_bytes = len(data_to_send_bytes)
    print(f"Attempting to send {total_bytes} bytes of {message_type} via LoRa.")

    if total_bytes > MAX_LORA_PAYLOAD_BYTES:
        print(
            f"ERROR: {message_type.capitalize()} data ({total_bytes} bytes) exceeds maximum LoRa payload size ({MAX_LORA_PAYLOAD_BYTES} bytes) for a single packet."
        )
        print(
            "Transmission aborted. You will need to re-implement chunking for larger data or reduce source size."
        )
        return

    try:
        print(f"Sending {total_bytes} bytes of {message_type} via RFM9x...")
        rfm9x.send(data_to_send_bytes)
        print(f"{message_type.capitalize()} sent successfully via LoRa radio!")

    except Exception as e:
        print(f"Error sending {message_type} via RFM9x radio: {e}")


def main_loop():
    print("Listening for incoming serial data...")

    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").strip()
                if line:
                    print(f"Received serial input: '{line}'")

                    if line == "CAPTURE_IMAGE":
                        print("Transmitting image via LoRa...")
                        encoded_data_string = capture_and_encode_image()
                        if encoded_data_string:
                            send_lora_message(encoded_data_string, "image")
                        else:
                            print(
                                "Skipping LoRa transmission of image due to previous errors."
                            )
                    elif line.startswith("telem-"):
                        print("Recieved telemetry... Transmitting via LoRa.")
                        cpuTemp = get_cpu_temp()
                        freeMem = psutil.virtual_memory().available
                        data = line.removeprefix("telem-").split(",")
                        data.append(get_cpu_temp())
                        data.append(freeMem)
                        encodedData = datapacker.pack(*data)
                        send_lora_message(
                            base64.b64encode(encodedData).decode("utf-8"), "telemetry"
                        )
                    else:
                        print(
                            f"Unrecognized serial input or invalid Base64 data: '{line}'. Ignoring."
                        )
            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            time.sleep(1)


if __name__ == "__main__":
    try:
        initialize_hardware()
        main_loop()
    except KeyboardInterrupt:
        print("\nApplication terminated by user (Ctrl+C).")
    finally:
        cleanup_hardware()
        print("Application gracefully exited.")
