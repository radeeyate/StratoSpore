package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"github.com/google/open-location-code/go"
)

const (
	outputDir            = "outputs_go"
	compressedFileName   = "compressed_image.bin"
	decompressedFileName = "decompressed.png"
)

var (
	currentCustomEncodedBytes []byte
	imageMutex                sync.RWMutex
	imageUpdateChan           = make(chan struct{}, 1)
	database                  *mongo.Database
	logs                      *mongo.Collection
	ctx                       = context.Background()
)

type Telemetry struct {
	ID                     primitive.ObjectID `bson:"_id,omitempty"`
	Altitude               float64            `bson:"altitude"`
	Time                   int64              `bson:"time"`
	Speed                  float64            `bson:"speed"`
	PlusCode               string             `bson:"plusCode"`
	HeatingPadTemp         float64            `bson:"heatingPadTemp"`
	OutsideTemp            float64            `bson:"outsideTemp"`
	Humidity               float64            `bson:"humidity"`
	FluorescenceRaw        int64              `bson:"fluorescenceRaw"`
	FluorescenceIrradiance float64            `bson:"fluorescenceIrradiance"`
	PicoTemp               float64            `bson:"picoTemp"`
	PicoMem                int64              `bson:"picoMem"`
	PiTemp                 float64            `bson:"piTemp"`
	PiMem                  int64              `bson:"piMem"`
	RxTime                 int64              `bson:"rxTime"`
	Latitude               float64            `bson:"latitiude"`
	Longitude              float64            `bson:"longitude"`
}

type UploadRequest struct {
	EncodedImage string `json:"encodedImage"`
}

func main() {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory %s: %v", outputDir, err)
	}

	connect()

	app := fiber.New()

	app.Use(logger.New())
	app.Use(cors.New())

	app.Post("/upload", handleUpload)
	app.Get("/stream", handleStream)
	app.Post("/telem", handlePostTelemetry)
	app.Get("/latest-telem", handleGetLatestTelemetry)

	app.Get("/", func(c *fiber.Ctx) error {
		c.Set(fiber.HeaderContentType, fiber.MIMETextHTML)
		return c.SendString(htmlClientPage)
	})

	log.Println("Server starting on :3000...")
	log.Fatal(app.Listen(":3000"))
}

func connect() {
	clientOptions := options.Client().
		ApplyURI("mongodb://localhost:27017")
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatal(err)
	}

	err = client.Ping(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}

	database = client.Database("stratospore-ground-testing")

	logs = database.Collection("logs")
}

func handlePostTelemetry(c *fiber.Ctx) error {
	altitudeStr := c.Query("altitude")
	fmt.Println("Received altitude:", altitudeStr)
	txTimeStr := c.Query("time")
	speedStr := c.Query("speed")
	plusCode := c.Query("plusCode")
	heatingPadTempStr := c.Query("heatingPadTemp")
	outsideTempStr := c.Query("outsideTemp")
	humidityStr := c.Query("humidity")
	fluorescenceRawStr := c.Query("fluorescenceRaw")
	fluorescenceIrradianceStr := c.Query("fluorescenceIrr")
	picoTempStr := c.Query("picoTemp")
	picoMemStr := c.Query("picoMem")
	piTempStr := c.Query("piTemp")
	piMemStr := c.Query("piMem")
	rxTime := time.Now().Unix()

	altitude, err := parseFloat(altitudeStr)
	if err != nil {
		log.Printf("Failed to parse altitude: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid altitude format"})
	}
	txTime, err := parseInt(txTimeStr)
	if err != nil {
		log.Printf("Failed to parse time: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid time format"})
	}
	speed, err := parseFloat(speedStr)
	if err != nil {
		log.Printf("Failed to parse speed: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid speed format"})
	}
	heatingPadTemp, err := parseFloat(heatingPadTempStr)
	if err != nil {
		log.Printf("Failed to parse heatingPadTemp: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid heatingPadTemp format"})
	}
	outsideTemp, err := parseFloat(outsideTempStr)
	if err != nil {
		log.Printf("Failed to parse outsideTemp: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid outsideTemp format"})
	}
	humidity, err := parseFloat(humidityStr)
	if err != nil {
		log.Printf("Failed to parse humidity: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid humidity format"})
	}
	fluorescenceRaw, err := parseInt(fluorescenceRawStr)
	if err != nil {
		log.Printf("Failed to parse fluorescenceRaw: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid fluorescenceRaw format"})
	}
	fluorescenceIrradiance, err := parseFloat(fluorescenceIrradianceStr)
	if err != nil {
		log.Printf("Failed to parse fluorescenceIrradiance: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid fluorescenceIrradiance format"})
	}
	picoTemp, err := parseFloat(picoTempStr)
	if err != nil {
		log.Printf("Failed to parse picoTemp: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid picoTemp format"})
	}
	picoMem, err := parseInt(picoMemStr)
	if err != nil {
		log.Printf("Failed to parse picoMem: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid picoMem format"})
	}
	piTemp, err := parseFloat(piTempStr)
	if err != nil {
		log.Printf("Failed to parse piTemp: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid piTemp format"})
	}
	piMem, err := parseInt(piMemStr)
	if err != nil {
		log.Printf("Failed to parse piMem: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid piMem format"})
	}

	location, err := olc.Decode(plusCode)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "bad pluscode"})
	}
	latitude, longitude := location.Center()
	fmt.Printf("Latitude: %f, Longitude: %f\n", latitude, longitude)

	telemetryData := Telemetry{
		Altitude:               altitude,
		Time:                   txTime,
		PlusCode:               plusCode,
		Speed:                  speed,
		HeatingPadTemp:         heatingPadTemp,
		OutsideTemp:            outsideTemp,
		Humidity:               humidity,
		FluorescenceRaw:        fluorescenceRaw,
		FluorescenceIrradiance: fluorescenceIrradiance,
		PicoTemp:               picoTemp,
		PicoMem:                picoMem,
		PiTemp:                 piTemp,
		PiMem:                  piMem,
		RxTime:                 rxTime,
		Latitude:               latitude,
		Longitude:              longitude,
	}

	_, err = logs.InsertOne(ctx, telemetryData)
	if err != nil {
		log.Printf("Failed to insert telemetry data: %v", err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to save telemetry data"})
	}

	log.Printf("Telemetry received and saved")
	return c.Status(fiber.StatusOK).JSON(fiber.Map{"message": "Telemetry received and saved"})
}

func handleGetLatestTelemetry(c *fiber.Ctx) error {
	var latestTelemetry Telemetry
	opts := options.FindOne().SetSort(bson.D{{Key: "rxTime", Value: -1}})
	err := logs.FindOne(ctx, bson.D{}, opts).Decode(&latestTelemetry)
	if err == mongo.ErrNoDocuments {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "No telemetry data found"})
	} else if err != nil {
		log.Printf("Failed to fetch latest telemetry data: %v", err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to fetch telemetry data"})
	}

	return c.Status(fiber.StatusOK).JSON(latestTelemetry)
}

func parseFloat(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}

func parseInt(s string) (int64, error) {
	return strconv.ParseInt(s, 10, 64)
}

func handleUpload(c *fiber.Ctx) error {
	var req UploadRequest
	if err := c.BodyParser(&req); err != nil {
		log.Printf("Error parsing upload request body: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Cannot parse request body"})
	}

	if req.EncodedImage == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "encodedImage field is required"})
	}

	customEncodedBytes, err := base64.StdEncoding.DecodeString(req.EncodedImage)
	if err != nil {
		log.Printf("Error base64 decoding image: %v", err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid base64 data"})
	}

	compressedFilePath := fmt.Sprintf("%s/%s", outputDir, compressedFileName)
	decompressedFilePath := fmt.Sprintf("%s/%s", outputDir, decompressedFileName)

	if err := os.WriteFile(compressedFilePath, customEncodedBytes, 0644); err != nil {
		log.Printf("Error writing compressed data to %s: %v", compressedFilePath, err)
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "Failed to save compressed image"})
	}

	cmd := exec.Command("./compig", "decode", "-input", compressedFilePath, "-output", decompressedFilePath)
	validationCmdOutput, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Error validating uploaded custom data via compig: %v. Output: %s", err, string(validationCmdOutput))
		errorDetail := fmt.Sprintf("compig validation failed: %v. Output: %s", err, string(validationCmdOutput))
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid custom image data format", "details": errorDetail})
	}

	info, err := os.Stat(decompressedFilePath)
	if err != nil {
		log.Printf("Validation: compig succeeded but output file '%s' not found or inaccessible: %v", decompressedFilePath, err)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid custom image data (compig validation produced no output file or it's inaccessible)"})
	}
	if info.Size() == 0 {
		log.Printf("Validation: compig succeeded but output file '%s' is empty.", decompressedFilePath)
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "Invalid custom image data (compig validation produced an empty file)"})
	}

	imageMutex.Lock()
	currentCustomEncodedBytes = customEncodedBytes
	imageMutex.Unlock()

	select {
	case imageUpdateChan <- struct{}{}:
	default:
	}

	log.Println("New image uploaded and validated successfully via compig.")
	return c.Status(fiber.StatusOK).JSON(fiber.Map{"message": "Image uploaded successfully"})
}

func handleStream(c *fiber.Ctx) error {
	c.Set("Content-Type", "text/event-stream")
	c.Set("Cache-Control", "no-cache")
	c.Set("Connection", "keep-alive")

	requestCtx := c.Context()

	requestCtx.SetBodyStreamWriter(func(w *bufio.Writer) {
		log.Println("SSE: Client connected. Starting stream writer.")
		defer log.Println("SSE: Stream writer finished for client.")

		if !sendCurrentImageAsStream(w) {
			return
		}

		for {
			select {
			case <-imageUpdateChan:
				log.Println("SSE: Image update signal received. Sending new image.")
				if !sendCurrentImageAsStream(w) {
					log.Println("SSE: Error sending image or writer closed during update.")
					return
				}
			case <-requestCtx.Done():
				log.Println("SSE: Client disconnected (detected by fasthttp context).")
				return
			}
		}
	})
	return nil
}

func sendCurrentImageAsStream(w *bufio.Writer) bool {
	imageMutex.RLock()
	isImageAvailable := currentCustomEncodedBytes != nil
	imageMutex.RUnlock()

	if !isImageAvailable {
		return true
	}

	compressedFilePath := fmt.Sprintf("%s/%s", outputDir, compressedFileName)
	decompressedFilePath := fmt.Sprintf("%s/%s", outputDir, decompressedFileName)

	cmd := exec.Command("./compig", "decode", "-input", compressedFilePath, "-output", decompressedFilePath)
	cmdCombinedOutput, err := cmd.CombinedOutput()
	if err != nil {
		errorMsg := fmt.Sprintf("Failed to decompress image using compig: %v. Output: %s", err, string(cmdCombinedOutput))
		log.Printf("SSE: %s", errorMsg)
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", errorMsg)
		if errFlush := w.Flush(); errFlush != nil {
			log.Printf("SSE: Error flushing (compig error event): %v", errFlush)
			return false
		}
		return true
	}

	pngBytes, err := os.ReadFile(decompressedFilePath)
	if err != nil {
		log.Printf("SSE: Error reading decompressed PNG file '%s': %v", decompressedFilePath, err)
		fmt.Fprintf(w, "event: error\ndata: Failed to read decompressed image from server: %s\n\n", err.Error())
		if errFlush := w.Flush(); errFlush != nil {
			log.Printf("SSE: Error flushing (read decompressed error event): %v", errFlush)
			return false
		}
		return true
	}

	if len(pngBytes) == 0 {
		log.Printf("SSE: Decompressed PNG file '%s' is empty.", decompressedFilePath)
		fmt.Fprintf(w, "event: error\ndata: Decompressed image is empty.\n\n")
		if errFlush := w.Flush(); errFlush != nil {
			log.Printf("SSE: Error flushing (empty decompressed file error event): %v", errFlush)
			return false
		}
		return true
	}

	base64PNG := base64.StdEncoding.EncodeToString(pngBytes)

	if _, err := fmt.Fprintf(w, "event: newImage\ndata: %s\n\n", base64PNG); err != nil {
		log.Printf("SSE: Error writing image data to stream writer: %v", err)
		return false
	}
	if err := w.Flush(); err != nil {
		log.Printf("SSE: Error flushing image data: %v", err)
		return false
	}

	log.Println("SSE: Sent image update to client (processed by compig).")
	return true
}

const htmlClientPage = `
<!DOCTYPE html>
<html>
<head>
    <title>Image and Telemetry Stream</title>
   <style>
    body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; margin-top: 20px; margin-bottom: 20px;}
    #imageContainer {
        border: 1px solid #ccc;
        width: 50vw;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
        background-color: #f0f0f0;
        overflow: hidden;
    }
    #streamedImage {
        width: 100%;
        height: 100%;
        object-fit: contain;
        image-rendering: pixelated;
        display: block;
    }
    textarea { width: 80%; min-height: 100px; margin-bottom: 10px; }
    button { padding: 10px 20px; }
    #status { margin-top: 10px; font-style: italic; }
    #telemetryContainer {
        border: 1px solid #ccc;
        padding: 15px;
        margin-top: 20px;
        width: 50vw;
        background-color: #e9e9e9;
    }
    #telemetryContainer h2 {
        margin-top: 0;
        color: #333;
    }
    #telemetryData p {
        margin: 5px 0;
    }
</style>
</head>
<body>
    <h1>Live Image Stream</h1>
    <div id="imageContainer">
        <img id="streamedImage" src="" alt="Waiting for image..." />
    </div>
    <div id="status"></div>

    <hr>

    <div id="telemetryContainer">
        <h2>Latest Telemetry Data</h2>
        <div id="telemetryData">
            <p><strong>Altitude:</strong> <span id="altitude">N/A</span> m</p>
            <p><strong>Time Sent:</strong> <span id="txTime">N/A</span></p>
            <p><strong>Plus Code:</strong> <span id="plusCode">N/A</span></p>
            <p><strong>Heating Pad Temp:</strong> <span id="heatingPadTemp">N/A</span> &deg;C</p>
            <p><strong>Outside Temp:</strong> <span id="outsideTemp">N/A</span> &deg;C</p>
            <p><strong>Humidity:</strong> <span id="humidity">N/A</span> %</p>
            <p><strong>Fluorescence Raw (ADC Count):</strong> <span id="fluorescenceRaw">N/A</span></p>
            <p><strong>Fluorescence Irradiance:</strong> <span id="fluorescenceIrradiance">N/A</span> uW/cm^2</p>
            <p><strong>Orpheus Pico Temperature:</strong> <span id="picoTemp">N/A</span> &deg;C</p>
            <p><strong>Orpheus Pico Memory Free:</strong> <span id="picoMem">N/A</span> bytes</p>
            <p><strong>Raspberry Pi Zero 2 W Temperature:</strong> <span id="piTemp">N/A</span> &deg;C</p>
            <p><strong>Raspberry Pi Zero 2 W Memory Free:</strong> <span id="piMem">N/A</span> bytes</p>
            <p><strong>Time Received:</strong> <span id="rxTime">N/A</span></p>
        </div>
        <div id="telemetryStatus" style="font-style: italic; margin-top: 10px;">Loading telemetry...</div>
    </div>

    <script>
        const imageElement = document.getElementById('streamedImage');
        const statusElement = document.getElementById('status');
        const telemetryStatusElement = document.getElementById('telemetryStatus');

        const eventSource = new EventSource('/stream');

		var lastUpdated = new Date().toLocaleTimeString();

        eventSource.addEventListener('newImage', function(event) {
            console.log('Received newImage event');
            const base64PNG = event.data;
            imageElement.src = 'data:image/png;base64,' + base64PNG;
            imageElement.alt = 'Streamed Image';
			lastUpdated = new Date().toLocaleTimeString()
            statusElement.textContent = 'Image updated at ' + lastUpdated;
        });
        
        eventSource.addEventListener('noImage', function(event) {
            console.log('Received noImage event:', event.data);
            imageElement.src = "";
            imageElement.alt = event.data || 'No image available yet.';
            statusElement.textContent = event.data || 'Waiting for first image upload.';
        });

        eventSource.addEventListener('error', function(event) {
            if (event.data) {
                 statusElement.textContent = 'SSE Error: ' + event.data;
                 console.error('SSE error from server:', event.data);
            } else if (event.target.readyState === EventSource.CLOSED) {
                statusElement.textContent = 'SSE connection closed by server or network error.';
                console.error('SSE connection closed. Event:', event);
                eventSource.close();
            } else if (event.target.readyState === EventSource.CONNECTING) {
                statusElement.textContent = 'SSE connection lost. Reconnecting...';
                console.log('SSE connection lost. Reconnecting... Event:', event);
            } else {
                statusElement.textContent = 'An unknown SSE error occurred.';
                console.error('Unknown SSE error:', event);
            }
        });

        function fetchLatestTelemetry() {
            fetch('/latest-telem')
                .then(response => {
                    if (!response.ok) {
                        if (response.status === 404) {
                            throw new Error('No telemetry data available yet.');
                        }
                        return response.json().then(errData => {
                            throw new Error(errData.error || 'Failed to fetch telemetry: ' + response.statusText);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    document.getElementById('altitude').textContent = data.Altitude !== undefined ? data.Altitude.toFixed(2) : 'N/A';
                    document.getElementById('txTime').textContent = data.Time !== undefined ? formatRelativeTime(data.Time) : 'N/A';
                    document.getElementById('plusCode').textContent = data.PlusCode || 'N/A';
                    document.getElementById('heatingPadTemp').textContent = data.HeatingPadTemp !== undefined ? data.HeatingPadTemp.toFixed(2) : 'N/A';
                    document.getElementById('outsideTemp').textContent = data.OutsideTemp !== undefined ? data.OutsideTemp.toFixed(2) : 'N/A';
                    document.getElementById('humidity').textContent = data.Humidity !== undefined ? data.Humidity.toFixed(2) : 'N/A';
                    document.getElementById('fluorescenceRaw').textContent = data.FluorescenceRaw !== undefined ? data.FluorescenceRaw : 'N/A';
                    document.getElementById('fluorescenceIrradiance').textContent = data.FluorescenceIrradiance !== undefined ? data.FluorescenceIrradiance.toFixed(2) : 'N/A';
                    document.getElementById('picoTemp').textContent = data.PicoTemp !== undefined ? data.PicoTemp.toFixed(2) : 'N/A';
                    document.getElementById('picoMem').textContent = data.PicoMem !== undefined ? data.PicoMem : 'N/A';
                    document.getElementById('piTemp').textContent = data.PiTemp !== undefined ? data.PiTemp.toFixed(2) : 'N/A';
                    document.getElementById('piMem').textContent = data.PiMem !== undefined ? data.PiMem : 'N/A';
                    document.getElementById('rxTime').textContent = data.RxTime !== undefined ? new Date(data.RxTime * 1000).toLocaleString() : 'N/A';
                    
                    telemetryStatusElement.textContent = 'Telemetry updated at ' + new Date().toLocaleTimeString();
                    console.log('Telemetry updated:', data);
                })
                .catch(error => {
                    console.error('Error fetching telemetry:', error);
                    telemetryStatusElement.textContent = 'Error fetching telemetry: ' + error.message;
                    document.getElementById('altitude').textContent = 'N/A';
                    document.getElementById('txTime').textContent = 'N/A';
                    document.getElementById('plusCode').textContent = 'N/A';
                    document.getElementById('heatingPadTemp').textContent = 'N/A';
                    document.getElementById('outsideTemp').textContent = 'N/A';
                    document.getElementById('humidity').textContent = 'N/A';
                    document.getElementById('fluorescenceRaw').textContent = 'N/A';
                    document.getElementById('fluorescenceIrradiance').textContent = 'N/A';
                    document.getElementById('picoTemp').textContent = 'N/A';
                    document.getElementById('picoMem').textContent = 'N/A';
                    document.getElementById('piTemp').textContent = 'N/A';
                    document.getElementById('piMem').textContent = 'N/A';
                    document.getElementById('rxTime').textContent = 'N/A';
                });
        }

		function formatRelativeTime(timestamp) {
  const now = Date.now();
  const diff = now - timestamp * 1000;
  const rtf = new Intl.RelativeTimeFormat('en', { style: 'short' });

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (seconds < 60) {
    return rtf.format(-seconds, 'second');
  } else if (minutes < 60) {
    return rtf.format(-minutes, 'minute');
  } else if (hours < 24) {
    return rtf.format(-hours, 'hour');
  } else {
    return rtf.format(-days, 'day');
  }
}


        setInterval(fetchLatestTelemetry, 500); 
		setInterval(() => {
			statusElement.textContent = 'Image updated at ' + lastUpdated + "; Current time: " + new Date().toLocaleTimeString();
		})
        fetchLatestTelemetry();

    </script>
</body>
</html>
`
