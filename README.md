# StratoSpore

StratoSpore is a payload which attempts to detect biological altitude utilizing chlorophyll fluorescence. 

## The Experiment

- Measure the fluorescence of chlorophyll in algae using photodiodes and light  
- Measure other stuff (e.g. altitude, geographical location, temperature, UV exposure, etc.) using sensors, and correlate the data with the photodiode output

## The Hypothesis

Chlorophyll fluorescence in algae will exhibit measurable changes in intensity or characteristics that correlate with variations in altitude and other environmental factors (temperature, UV exposure, etc.).

When exposed to specific wavelengths of light, chlorophyll fluoresces in a telltale red signature. In fact, a very common high and middle school experiment known as bloody chlorophyll, leverages this effect to produce a haunting blood like color from the otherwise green plant slurry.

Chlorophyll fluorescence usually peaks around 680-685 nm. Using a bandpass filter, we selectively let light into a viewing chamber. This selection yields us a readable level of clean fluorescence which can be measured using an optical sensor made up of photodiodes and read using the Raspberry Pi Pico 2’s (or Orpheus Pico) UART interface. Previously, we planned on just using a normal photodiode, but found an Integrated Circuit that would be much less expensive and easier to use.

---

## **Hardware and Electrical**

The payload will consist of 3 PCBs, which will collect, process, and transmit data. We plan on using a Raspberry Pi Zero 2 WH for the main data recording and transmission. If possible, we hope to transmit all recorded data live over a 915Mhz LoRa signal. All sensor data will be collected by a Raspberry Pi Pico 2 (or Orpheus Pico) before being fed into the Pi Zero 2 WH over UART.

## **Data Collection and Processing**

The Raspberry Pi Pico 2 will be responsible for interfacing with various sensors, including:

- [x] GPS module for geographical location  
- [x] Temperature, barometric pressure, and humidity sensor  
- [x] UV exposure sensor  
- [x] Photodiode for chlorophyll fluorescence measurement  
- [x] Thermocouple for checking heating pad temperature  
- [x] Temperature sensor for algae

### BME280 Circuit

To have data we can correlate with the fluorescence levels, we will be using a **BME280** to gather certain environmental properties including temperature, humidity, and barometric pressure. It will communicate with the Raspberry Pi Pico 2 (or Orpheus Pico) via its I2C interface.

### UV Sensor

We are using Adafruit’s Analog UV Light Sensor Breakout to measure UV index using a photodiode. Originally, we made our own circuit using Adafruit’s schematics, but later decided to save the money by just buying the breakouts. It will be connected to the Raspberry Pi Pico 2’s (or Orpheus Pico) ADC. It will measure UV light in the 240-370 nm range which covers UVB and most of the UVA spectrum. UV index will be measured by dividing the output voltage by 0.1. If the output voltage is 0.5V, the UV index is 5\. It will be directly soldered onto the sensor board, mounted outside the payload.

### Fluorescence Measurement Board

A PCB will be mounted to the flask of algae so we can receive the most accurate reading of Relative Fluorescence Units. An **AS7263-BLGT** sensor is being used because it can capture visible light at 680 nm, which is the peak for chlorophyll fluorescence. It will communicate with the Raspberry Pi Pico 2 (or Orpheus Pico) over UART.  

### GPS

We will be using a Neo 6M GPS breakout from Amazon.

Data from these sensors will be collected at regular intervals and timestamped. The Pico 2 (or Orpheus Pico) will then process this data and transmit it to the Raspberry Pi Zero 2 WH via UART.

The Raspberry Pi Zero 2 WH will act as the central hub for data logging and transmission. It will:

- [ ] Receive data from the Pico 2 (or Orpheus Pico) over UART, I2C, and ADC  
- [ ] Store the data locally  
- [ ] Attempt to transmit the data live over LoRa  
- [ ] Potentially perform additional data analysis and prediction

All data will be logged with InfluxDB, and viewed as a Grafana dashboard

## **Power Supply**

The payload will require a reliable power source. We will be using:

- [x] 8x Energizer Ultimate Lithium AA Batteries

A power management circuit will be essential to regulate the voltage and ensure a stable power supply to all components. As we require many components, we must make two voltage rails so we can provide the correct power to the correct components. We will have the following voltage rails:

- [x] \+5V \- **for various components including the Raspberry Pi Zero 2 WH**  
      - A regulator is used for simplicity and cost sake  
- [x] \+3.3V \- **for various sensors**  
      - A regulator is also used

## **Enclosure**

The payload will need to be housed in a protective enclosure. The enclosure must:

- [ ] Be lightweight  
- [ ] Be durable enough to withstand the conditions of high altitude flight  
- [ ] Protect the electronics from temperature extremes and UV radiation  
- [ ] Mount a sensor board on the top  
- [ ] Have standoffs for mounting PCBs and more on the bottom  
- [ ] Have a window to let light in  
- [ ] Have a camera to take pictures

### The Window

We will need light to come in through an acrylic window so our algae can be exposed to light to make it photosynthesize and fluoresce. In front of the window will also be a Raspberry Pi Camera Module 3 to take pictures.  

The box will contain the mainboard with the Pico (or Orpheus Pico) and power supplies, a Raspberry Pi Zero 2 W with a 915 Mhz LoRa module, the camera, fluorescence sensor (+ temperature), and the sensor board mounted on the top of the box.  

## Current Tasks

| Description | Status | Notes |
| :---- | :---- | :---- |
| Finish Power Supply Design | In Progress |  |
| Order PCBs and Assemble | Not Started |  |
| Create Firmware | Not Started |  |
| Finish CAD of Box | In Progress |  |
| Create and print standoff platform | Not Started |  |
| Order algae and grow more of it | Not Started |  |
| Test everything | Not Started |  |

Some parts will be purchased personally. This list does also not include all purchases needed, and will be updated if needed.	

References

Fluorescence. [https://www.esa.int/Applications/Observing\_the\_Earth/FutureEO/FLEX/Fluorescence](https://www.esa.int/Applications/Observing_the_Earth/FutureEO/FLEX/Fluorescence). Accessed 22 Mar. 2025\.
