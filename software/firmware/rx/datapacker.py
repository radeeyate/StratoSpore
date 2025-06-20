import struct
import base64

_STRUCT_FORMAT = ">fl11shhhhhhhhIII"


def pack(
    altitude,
    time,
    location,
    speed,
    uvIndex,
    heatingPadTemp,
    outsideTemp,
    humidity,
    fluorescence,
    picoTemp,
    picoMem,
    piTemp,
    piMem,
):
    data = (
        float(altitude),
        int(time),
        location.encode("utf-8"),
        int(float(speed) * 10),
        int(float(uvIndex) * 10),
        int(round(float(heatingPadTemp), 2) * 100),
        int(round(float(outsideTemp), 2) * 100),
        int(round(float(humidity), 1) * 10),
        int(fluorescence),
        int(round(float(picoTemp), 1) * 10),
        int(picoMem),
        int(round(float(piTemp), 1) * 10),
        int(piMem),
    )

    print(data)

    return struct.pack(_STRUCT_FORMAT, *data)


def unpack(data):
    data = list(struct.unpack(_STRUCT_FORMAT, data))

    data[2] = data[2].decode("utf-8").strip("\x00")  # decode location plus code
    data[3] = data[3] / 10  # speed (kph)
    data[4] = data[4] / 10  # uv index
    data[5] = data[5] / 100  # heating pad temp
    data[6] = data[6] / 100  # ambient temp
    data[7] = data[7] / 10  # humidity
    data[9] = data[9] / 10  # pico temp
    data[10] = data[10] / 10  # pi temp

    return tuple(data)
