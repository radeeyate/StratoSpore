package main

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg" // Register JPEG decoder
	"image/png"    // Register PNG decoder
	"io"
	"log"
	"math"
	"os"
	"sort"

	"github.com/disintegration/gift"
	"github.com/nfnt/resize"
)

const (
	TargetWidth             uint  = 27
	TargetHeight            uint  = 15
	NumPaletteColors        int   = 4
	BlurSigma               float32 = 0.8
	QuantizationShiftAmount uint  = 5
)

type RGB struct {
	R, G, B uint8
}

type CompressionStats struct {
	InitialSizeBytes         int64
	SizeBeforeZlibBytes    int
	FinalCompressedSizeBytes int64
	CompressionRatio         float64
	SpaceSavedPercentage     float64
}

func calculateLuminance(c RGB) float32 {
	return 0.299*float32(c.R) + 0.587*float32(c.G) + 0.114*float32(c.B)
}

func quantizeColor(c RGB, shiftAmount uint) RGB {
	rQuant := (c.R >> shiftAmount) << shiftAmount
	gQuant := (c.G >> shiftAmount) << shiftAmount
	bQuant := (c.B >> shiftAmount) << shiftAmount

	return RGB{rQuant, gQuant, bQuant}
}

func generateDynamicPalette(img *image.RGBA, numColors int, quantShift uint) []RGB {
	colorCounts := make(map[RGB]int)
	bounds := img.Bounds()

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			originalColor := RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
			quantized := quantizeColor(originalColor, quantShift)
			colorCounts[quantized]++
		}
	}

	type colorCountPair struct {
		Color RGB
		Count int
	}

	var sortedColors []colorCountPair
	for color, count := range colorCounts {
		sortedColors = append(sortedColors, colorCountPair{color, count})
	}

	sort.Slice(sortedColors, func(i, j int) bool {
		return sortedColors[i].Count > sortedColors[j].Count
	})

	palette := make([]RGB, 0, numColors)
	for i := 0; i < len(sortedColors) && i < numColors; i++ {
		palette = append(palette, sortedColors[i].Color)
	}

	sort.Slice(palette, func(i, j int) bool {
		lumA := calculateLuminance(palette[i])
		lumB := calculateLuminance(palette[j])
		return lumA > lumB
	})

	for len(palette) < numColors {
		log.Printf("Padding palette with black as it has %d colors, expected %d", len(palette), numColors)
		palette = append(palette, RGB{0, 0, 0})
	}
	if len(palette) > numColors {
		palette = palette[:numColors]
	}

	return palette
}

func findClosestPaletteIndex(color RGB, palette []RGB) uint8 {
	minDist := uint32(math.MaxUint32)
	bestIdx := uint8(0)

	for i, pColor := range palette {
		dr := int32(color.R) - int32(pColor.R)
		dg := int32(color.G) - int32(pColor.G)
		db := int32(color.B) - int32(pColor.B)
		dist := uint32(dr*dr + dg*dg + db*db)

		if dist < minDist {
			minDist = dist
			bestIdx = uint8(i)
		}
		if dist == 0 {
			break
		}
	}
	return bestIdx
}

func compressImage(inputPath, outputPath string) ([]RGB, CompressionStats, error) {
	log.Printf("Starting compression of %s...", inputPath)

	inputFile, err := os.Open(inputPath)
	if err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to open input file: %w", err)
	}
	defer inputFile.Close()

	inputInfo, err := inputFile.Stat()
	if err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to get input file info: %w", err)
	}
	initialSize := inputInfo.Size()
	log.Printf("Initial size: %d bytes", initialSize)

	img, _, err := image.Decode(inputFile)
	if err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to decode image: %w", err)
	}

	resizedImg := resize.Resize(TargetWidth, TargetHeight, img, resize.NearestNeighbor)
	log.Printf("Resized image to %dx%d", TargetWidth, TargetHeight)

	g := gift.New(gift.GaussianBlur(BlurSigma))
	blurredImg := image.NewRGBA(g.Bounds(resizedImg.Bounds()))
	g.Draw(blurredImg, resizedImg)
	log.Printf("Applied Gaussian blur with sigma: %f", BlurSigma)

	rgbaImg := image.NewRGBA(blurredImg.Bounds())
	draw.Draw(rgbaImg, rgbaImg.Bounds(), blurredImg, image.Point{}, draw.Src)

	dynamicPalette := generateDynamicPalette(rgbaImg, NumPaletteColors, QuantizationShiftAmount)
	log.Printf("Generated dynamic palette (%d colors):", len(dynamicPalette))
	for i, pCol := range dynamicPalette {
		log.Printf("%2d: RGB(%3d, %3d, %3d)", i, pCol.R, pCol.G, pCol.B)
	}

	bitsPerPixel := 0
	if NumPaletteColors > 0 {
		bitsPerPixel = int(math.Ceil(math.Log2(float64(NumPaletteColors))))
	}
	if bitsPerPixel == 0 && NumPaletteColors > 1 {
		bitsPerPixel = 1 
	} else if NumPaletteColors == 1 {
		bitsPerPixel = 1
	}


	if bitsPerPixel == 0 {
		return nil, CompressionStats{}, fmt.Errorf("calculated bitsPerPixel is 0 with %d palette colors", NumPaletteColors)
	}
	log.Printf("Using %d bits per pixel.", bitsPerPixel)


	var paletteIndexedData bytes.Buffer
	var bitBuffer uint32
	var bitsInBuffer uint8

	bounds := rgbaImg.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := rgbaImg.At(x, y).RGBA()
			pixelColor := RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
			paletteIndex := findClosestPaletteIndex(pixelColor, dynamicPalette)

			bitBuffer = (bitBuffer << uint(bitsPerPixel)) | uint32(paletteIndex)
			bitsInBuffer += uint8(bitsPerPixel)

			for bitsInBuffer >= 8 {
				byteToWrite := uint8(bitBuffer >> (bitsInBuffer - 8))
				paletteIndexedData.WriteByte(byteToWrite)
				bitsInBuffer -= 8
				if bitsInBuffer > 0 {
					bitBuffer &= (1 << bitsInBuffer) - 1
				} else {
					bitBuffer = 0
				}
			}
		}
	}

	if bitsInBuffer > 0 {
		byteToWrite := uint8(bitBuffer << (8 - bitsInBuffer))
		paletteIndexedData.WriteByte(byteToWrite)
	}

	sizeBeforeZlib := paletteIndexedData.Len()
	log.Printf("Size before zlib (palette-indexed): %d bytes", sizeBeforeZlib)

	var zlibCompressedData bytes.Buffer
	zlibWriter, _ := zlib.NewWriterLevel(&zlibCompressedData, zlib.BestCompression)
	if _, err := zlibWriter.Write(paletteIndexedData.Bytes()); err != nil {
		return nil, CompressionStats{}, fmt.Errorf("zlib compression failed: %w", err)
	}
	if err := zlibWriter.Close(); err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to close zlib writer: %w", err)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to create output file: %w", err)
	}
	defer outputFile.Close()

	if _, err := outputFile.Write(zlibCompressedData.Bytes()); err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to write compressed data: %w", err)
	}

	outputInfo, err := outputFile.Stat()
	if err != nil {
		return nil, CompressionStats{}, fmt.Errorf("failed to get output file info: %w", err)
	}
	compressedSize := outputInfo.Size()
	log.Printf("Compressed size (after zlib): %d bytes", compressedSize)

	var compressionRatio, spaceSaved float64
	if initialSize > 0 && compressedSize > 0 {
		compressionRatio = float64(initialSize) / float64(compressedSize)
		spaceSaved = (1.0 - (float64(compressedSize) / float64(initialSize))) * 100.0
	} else {
		log.Println("Cannot calculate compression ratio (initial or compressed size is 0).")
	}
	log.Printf("Compressed image saved to %s", outputPath)

	stats := CompressionStats{
		InitialSizeBytes:         initialSize,
		SizeBeforeZlibBytes:    sizeBeforeZlib,
		FinalCompressedSizeBytes: compressedSize,
		CompressionRatio:         compressionRatio,
		SpaceSavedPercentage:     spaceSaved,
	}
	return dynamicPalette, stats, nil
}

func decompressImage(inputPath, outputPath string, palette []RGB, targetW, targetH uint) error {
	log.Printf("Starting decompression of %s (saving to %s)...", inputPath, outputPath)

	compressedFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("failed to open compressed file: %w", err)
	}
	defer compressedFile.Close()
	
	compressedInfo, _ := compressedFile.Stat()
	log.Printf("Zlib compressed file size to decompress: %d bytes", compressedInfo.Size())


	zlibReader, err := zlib.NewReader(compressedFile)
	if err != nil {
		return fmt.Errorf("failed to create zlib reader: %w", err)
	}
	defer zlibReader.Close()

	paletteIndexedData, err := io.ReadAll(zlibReader)
	if err != nil {
		return fmt.Errorf("failed to decompress zlib data: %w", err)
	}
	log.Printf("Size after zlib decompression (palette-indexed): %d bytes", len(paletteIndexedData))

	bitsPerPixel := 0
	if NumPaletteColors > 0 {
		bitsPerPixel = int(math.Ceil(math.Log2(float64(NumPaletteColors))))
	}
	if bitsPerPixel == 0 && NumPaletteColors > 1 {
		bitsPerPixel = 1
	} else if NumPaletteColors == 1 {
		bitsPerPixel = 1
	}

	if bitsPerPixel == 0 {
		return fmt.Errorf("bitsPerPixel is 0 for decompression with %d palette colors", NumPaletteColors)
	}


	decompressedImg := image.NewRGBA(image.Rect(0, 0, int(targetW), int(targetH)))
	var bitBuffer uint32
	var bitsInBuffer uint8
	byteIdx := 0
	totalPixels := int(targetW * targetH)
	pixelsDecoded := 0

	pixelMask := uint32((1 << bitsPerPixel) - 1)

	for y := 0; y < int(targetH); y++ {
		for x := 0; x < int(targetW); x++ {
			if pixelsDecoded >= totalPixels {
				break
			}

			for bitsInBuffer < uint8(bitsPerPixel) {
				if byteIdx < len(paletteIndexedData) {
					bitBuffer = (bitBuffer << 8) | uint32(paletteIndexedData[byteIdx])
					bitsInBuffer += 8
					byteIdx++
				} else {
					if pixelsDecoded < totalPixels && bitsInBuffer < uint8(bitsPerPixel) {
						if bitsInBuffer == 0 && pixelsDecoded < totalPixels {
							return fmt.Errorf("unexpected end of compressed data. Decoded %d of %d pixels. byteIdx: %d, len: %d", pixelsDecoded, totalPixels, byteIdx, len(paletteIndexedData))
						}

						log.Printf("Warning: Not enough bits for a full pixel index at the end. bitsInBuf: %d, bitsPerPix: %d", bitsInBuffer, bitsPerPixel)
					}
					break
				}
			}

			if bitsInBuffer >= uint8(bitsPerPixel) {
				shiftAmount := bitsInBuffer - uint8(bitsPerPixel)
				paletteIndex := uint8((bitBuffer >> shiftAmount) & pixelMask)

				if int(paletteIndex) >= len(palette) {
					return fmt.Errorf("decoded palette index %d out of bounds for palette size %d", paletteIndex, len(palette))
				}

				pColor := palette[paletteIndex]
				decompressedImg.Set(x, y, color.RGBA{pColor.R, pColor.G, pColor.B, 255})
				pixelsDecoded++

				bitsInBuffer -= uint8(bitsPerPixel)
				if bitsInBuffer > 0 {
					bitBuffer &= (1 << bitsInBuffer) - 1
				} else {
					bitBuffer = 0
				}
			} else if pixelsDecoded < totalPixels {
				log.Printf("Warning: Insufficient bits remaining in buffer for pixel %d. Bits needed: %d, bits have: %d.", pixelsDecoded, bitsPerPixel, bitsInBuffer)
			}
		}
		if pixelsDecoded >= totalPixels {
			break
		}
	}

	if pixelsDecoded < totalPixels {
		log.Printf("Warning: Decoded only %d of %d pixels. Input file might be truncated or corrupted.", pixelsDecoded, totalPixels)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create decompressed output file: %w", err)
	}
	defer outputFile.Close()

	if err := png.Encode(outputFile, decompressedImg); err != nil {
		return fmt.Errorf("failed to encode decompressed image: %w", err)
	}
	log.Printf("Decompressed image saved to %s", outputPath)
	return nil
}

func decompressImageGrayscale(inputPath, outputPath string, targetW, targetH uint) error {
	log.Printf("Starting Grayscale decompression of %s (saving to %s)...", inputPath, outputPath)

	compressedFile, err := os.Open(inputPath)
	if err != nil {
		return fmt.Errorf("failed to open compressed file (grayscale): %w", err)
	}
	defer compressedFile.Close()
	
	compressedInfo, _ := compressedFile.Stat()
	log.Printf("Zlib compressed file size to decompress (grayscale): %d bytes", compressedInfo.Size())

	zlibReader, err := zlib.NewReader(compressedFile)
	if err != nil {
		return fmt.Errorf("failed to create zlib reader (grayscale): %w", err)
	}
	defer zlibReader.Close()

	paletteIndexedData, err := io.ReadAll(zlibReader)
	if err != nil {
		return fmt.Errorf("failed to decompress zlib data (grayscale): %w", err)
	}
	log.Printf("Size after zlib decompression (grayscale): %d bytes", len(paletteIndexedData))

	bitsPerPixel := 0
	if NumPaletteColors > 0 {
		bitsPerPixel = int(math.Ceil(math.Log2(float64(NumPaletteColors))))
	}
	if bitsPerPixel == 0 && NumPaletteColors > 1 {
		bitsPerPixel = 1
	} else if NumPaletteColors == 1 {
		bitsPerPixel = 1
	}


	if bitsPerPixel == 0 {
		return fmt.Errorf("bitsPerPixel is 0 for grayscale decompression with %d palette colors", NumPaletteColors)
	}

	decompressedImg := image.NewGray(image.Rect(0, 0, int(targetW), int(targetH)))
	var bitBuffer uint32
	var bitsInBuffer uint8
	byteIdx := 0
	totalPixels := int(targetW * targetH)
	pixelsDecoded := 0

	pixelMask := uint32((1 << bitsPerPixel) - 1)

	for y := 0; y < int(targetH); y++ {
		for x := 0; x < int(targetW); x++ {
			if pixelsDecoded >= totalPixels {
				break
			}

			for bitsInBuffer < uint8(bitsPerPixel) {
				if byteIdx < len(paletteIndexedData) {
					bitBuffer = (bitBuffer << 8) | uint32(paletteIndexedData[byteIdx])
					bitsInBuffer += 8
					byteIdx++
				} else {
					if pixelsDecoded < totalPixels && bitsInBuffer < uint8(bitsPerPixel) {
                         if bitsInBuffer == 0 && pixelsDecoded < totalPixels {
						    return fmt.Errorf("(Grayscale) Unexpected end of compressed data. Decoded %d of %d pixels", pixelsDecoded, totalPixels)
                         }
					}
					break
				}
			}

			if bitsInBuffer >= uint8(bitsPerPixel) {
				shiftAmount := bitsInBuffer - uint8(bitsPerPixel)
				paletteIndex := uint8((bitBuffer >> shiftAmount) & pixelMask)

				var grayscaleValue uint8
				if NumPaletteColors <= 1 {
					if paletteIndex == 0 {
						grayscaleValue = 0
					} else {
						grayscaleValue = 255
					}
				} else {
					grayscaleValue = 255 - uint8(math.Round(float64(paletteIndex)*255.0/float64(NumPaletteColors-1)))
				}

				decompressedImg.SetGray(x, y, color.Gray{Y: grayscaleValue})
				pixelsDecoded++

				bitsInBuffer -= uint8(bitsPerPixel)
				if bitsInBuffer > 0 {
					bitBuffer &= (1 << bitsInBuffer) - 1
				} else {
					bitBuffer = 0
				}
			} else if pixelsDecoded < totalPixels {
				log.Printf("Warning (Grayscale): Insufficient bits remaining for pixel %d.", pixelsDecoded)
			}
		}
		if pixelsDecoded >= totalPixels {
			break
		}
	}
	if pixelsDecoded < totalPixels {
		log.Printf("Warning (Grayscale): Decoded only %d of %d pixels.", pixelsDecoded, totalPixels)
	}

	outputFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create grayscale output file: %w", err)
	}
	defer outputFile.Close()

	if err := png.Encode(outputFile, decompressedImg); err != nil {
		return fmt.Errorf("failed to encode grayscale image: %w", err)
	}
	log.Printf("Grayscale decompressed image saved to %s", outputPath)
	return nil
}

func createSampleInputImage(filePath string) error {
	if _, err := os.Stat(filePath); err == nil {
		return nil
	}
	log.Printf("Creating a sample input image (%s)...", filePath)
	img := image.NewRGBA(image.Rect(0, 0, 256, 256))
	for x := 0; x < 256; x++ {
		for y := 0; y < 256; y++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), uint8((x + y) % 256), 255})
		}
	}
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create sample image file: %w", err)
	}
	defer file.Close()
	if err := png.Encode(file, img); err != nil {
		return fmt.Errorf("failed to encode sample image: %w", err)
	}
	log.Printf("%s created.", filePath)
	return nil
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	inputImagePath := "input.jpg"
	_ = os.Mkdir("outputs_go", 0755)
	compressedFilePath := "outputs_go/compressed_image.bin"
	decompressedImagePath := "outputs_go/decompressed_output.png"
	decompressedGrayscalePath := "outputs_go/decompressed_output_grayscale.png"

	if err := createSampleInputImage(inputImagePath); err != nil {
		log.Fatalf("Failed to ensure sample input image: %v", err)
	}

	palette, stats, err := compressImage(inputImagePath, compressedFilePath)
	if err != nil {
		log.Fatalf("Compression failed: %v", err)
	}

	err = decompressImage(compressedFilePath, decompressedImagePath, palette, TargetWidth, TargetHeight)
	if err != nil {
		log.Fatalf("Decompression failed: %v", err)
	}

	err = decompressImageGrayscale(compressedFilePath, decompressedGrayscalePath, TargetWidth, TargetHeight)
	if err != nil {
		log.Fatalf("Grayscale decompression failed: %v", err)
	}

	fmt.Println("\n--- Go Compression Report ---")
	fmt.Printf("Initial image size: %d bytes\n", stats.InitialSizeBytes)
	fmt.Printf("Size before zlib: %d bytes\n", stats.SizeBeforeZlibBytes)
	fmt.Printf("Size after zlib: %d bytes\n", stats.FinalCompressedSizeBytes)
	if stats.FinalCompressedSizeBytes > 0 {
		fmt.Printf("Overall compression ratio: %.2f:1\n", stats.CompressionRatio)
		fmt.Printf("Overall space saved: %.2f%%\n", stats.SpaceSavedPercentage)
	} else {
		fmt.Println("Overall compression ratio: N/A (final size is 0)")
		fmt.Println("Overall space saved: N/A")
	}
	fmt.Printf("Target dimensions: %dx%d\n", TargetWidth, TargetHeight)
	fmt.Printf("Palette colors used: %d (%.0f bpp)\n", NumPaletteColors, math.Ceil(math.Log2(float64(NumPaletteColors))))
	fmt.Printf("Quantization shift: %d\n", QuantizationShiftAmount)
	fmt.Printf("Blur sigma applied: %.2f\n", BlurSigma)
	fmt.Println("--- End of Report ---")

	log.Println("\nProcess completed.")
}