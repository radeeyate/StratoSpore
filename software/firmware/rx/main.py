import time
import busio
from digitalio import DigitalInOut, Direction
import board
import adafruit_rfm9x
import base64
import requests
import datapacker


CS = DigitalInOut(board.CE1)
RESET = DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
IP = "10.195.197.48"
SERVER_URL = f"http://{IP}:3000/upload"
TELEM_URL = f"http://{IP}:3000/telem"

try:
    rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 910.0)
    print("RFM9x: Detected and configured.")
    # rfm9x.receive_timeout = 0.5 maybe?
except RuntimeError as error:
    print("RFM9x Error: ", error)
    print("Please ensure the RFM9x module is correctly wired and powered.")
    exit(1)

rfm9x.tx_power = 23

rfm9x.spreading_factor = 7
rfm9x.frequency_mhz = 910.0
rfm9x.coding_rate = 6
rfm9x.low_datarate_optimize = True
rfm9x.enable_crc = True

CONVERSION_FACTOR_RED_UW_CM2 = 15.5 / 5435  # 0.002851886

print("Starting LoRa reception loop...")
print("Waiting for messages...")


def post_image_to_server(encoded_image_data):
    if not encoded_image_data:
        print("No image data to post.")
        return

    headers = {"Content-Type": "application/json"}
    payload = {"encodedImage": encoded_image_data}

    try:
        response = requests.post(SERVER_URL, json=payload, headers=headers)
        response.raise_for_status()
        print(f"Successfully posted image to server. Response: {response.json()}")
    except requests.exceptions.ConnectionError as e:
        print(
            f"Connection error: Could not connect to the server at {SERVER_URL}. Is the server running and accessible? Error: {e}"
        )
    except requests.exceptions.Timeout:
        print("The request timed out.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")


def post_telem_to_server(
    altitude,
    time,
    location,
    speed,
    uvIndex,
    heatingPadTemp,
    outsideTemp,
    humidity,
    fluorescenceRaw,
    picoTemp,
    picoMem,
    piTemp,
    piMem,
    fluorescenceIrr,
):
    headers = {"Content-Type": "application/json"}
    payload = {
        "altitude": altitude,
        "time": time,
        "plusCode": location,
        "speed": speed,
        "uvIndex": uvIndex,
        "heatingPadTemp": heatingPadTemp,
        "outsideTemp": outsideTemp,
        "humidity": humidity,
        "fluorescenceRaw": fluorescenceRaw,
        "fluorescenceIrr": round(fluorescenceIrr, 4),
        "picoTemp": picoTemp,
        "picoMem": picoMem,
        "piTemp": piTemp,
        "piMem": piMem,
    }
    print(payload)

    try:
        response = requests.post(TELEM_URL, params=payload, headers=headers)
        response.raise_for_status()
        print(f"Successfully posted telemetry to server. Response: {response.json()}")
    except requests.exceptions.ConnectionError as e:
        print(
            f"Connection error: Could not connect to the server at {TELEM_URL}. Is the server running and accessible? Error: {e}"
        )
    except requests.exceptions.Timeout:
        print("The request timed out.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"An error occured during the request: {e}")


while True:
    packet = rfm9x.receive()

    if packet is not None:
        try:
            packet_text = str(packet, "utf-8")
            print(len(packet_text))
            decodedText = base64.b64decode(packet_text)
            if len(decodedText) == 45:
                unpackedData = datapacker.unpack(decodedText)
                print(len(unpackedData))
                fluorescenceIrr = unpackedData[7] * CONVERSION_FACTOR_RED_UW_CM2
                post_telem_to_server(*unpackedData, fluorescenceIrr=fluorescenceIrr)
                print("Recieved telemetry!")
                continue

            print("Recieved image!")
            post_image_to_server(packet_text)

            print(f"  RSSI: {rfm9x.last_rssi} dB")
            if rfm9x.snr is not None:
                print(f"  SNR: {rfm9x.snr} dB")

        except UnicodeDecodeError:
            print(f"Received non-UTF-8 packet: {packet}")
            print(f"  RSSI: {rfm9x.last_rssi} dB")
            if rfm9x.snr is not None:
                print(f"  SNR: {rfm9x.snr} dB")
    else:
        pass

    time.sleep(0.1)