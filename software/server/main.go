package main

import (
	"bufio"
	"encoding/base64"
	"fmt"
	"log"
	"os"
	"os/exec"
	"sync"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
)

const (
	outputDir           = "outputs_go"
	compressedFileName  = "compressed_image.bin"
	decompressedFileName = "decompressed.png"
)

var (
	currentCustomEncodedBytes []byte
	imageMutex                sync.RWMutex
	imageUpdateChan           = make(chan struct{}, 1)
)

type UploadRequest struct {
	EncodedImage string `json:"encodedImage"`
}

func main() {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("Failed to create output directory %s: %v", outputDir, err)
	}

	app := fiber.New()

	app.Use(logger.New())
	app.Use(cors.New())

	app.Post("/upload", handleUpload)
	app.Get("/stream", handleStream)

	app.Get("/", func(c *fiber.Ctx) error {
		c.Set(fiber.HeaderContentType, fiber.MIMETextHTML)
		return c.SendString(htmlClientPage)
	})

	log.Println("Server starting on :3000...")
	log.Fatal(app.Listen(":3000"))
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
    <title>Image Stream</title>
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
</style>
</head>
<body>
    <h1>Live Image Stream</h1>
    <div id="imageContainer">
        <img id="streamedImage" src="" alt="Waiting for image..." />
    </div>

    <h2>Upload Custom Encoded Image (Base64)</h2>
    <textarea id="base64Input" placeholder="Paste your custom base64 encoded image data here..."></textarea>
    <button onclick="uploadImage()">Upload Image</button>
    <div id="status"></div>

    <script>
        const imageElement = document.getElementById('streamedImage');
        const statusElement = document.getElementById('status');

        const eventSource = new EventSource('/stream');

        eventSource.addEventListener('newImage', function(event) {
            console.log('Received newImage event');
            const base64PNG = event.data;
            imageElement.src = 'data:image/png;base64,' + base64PNG;
            imageElement.alt = 'Streamed Image';
            statusElement.textContent = 'Image updated at ' + new Date().toLocaleTimeString();
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

        function uploadImage() {
            const base64Data = document.getElementById('base64Input').value;
            if (!base64Data.trim()) {
                statusElement.textContent = 'Error: Base64 input cannot be empty.';
                return;
            }
            statusElement.textContent = 'Uploading...';

            fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ encodedImage: base64Data }),
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errData => { 
                        throw new Error(errData.details || errData.error || 'Upload failed: ' + response.statusText);
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Upload successful:', data);
                statusElement.textContent = 'Upload successful! Stream should update.';
                document.getElementById('base64Input').value = '';
            })
            .catch(error => {
                console.error('Upload error:', error);
                statusElement.textContent = 'Upload error: ' + error.message;
            });
        }
    </script>
</body>
</html>
`