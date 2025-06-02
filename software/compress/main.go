package main

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg" 
	"image/png"  
	"io"
	"log"
	"math"
	"os"
	"sort"

	"github.com/disintegration/gift"
	"github.com/nfnt/resize"
)

const (
	TargetWidth             uint    = 27
	TargetHeight            uint    = 15
	NumPaletteColors        int     = 4
	BlurSigma               float32 = 0.8
	QuantizationShiftAmount uint    = 5
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

func generateDynamicPalette(img *image.RGBA, numColorsTarget int, quantShift uint) []RGB {
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

	palette := make([]RGB, 0, numColorsTarget)
	for i := 0; i < len(sortedColors) && i < numColorsTarget; i++ {
		palette = append(palette, sortedColors[i].Color)
	}

	sort.Slice(palette, func(i, j int) bool {
		lumA := calculateLuminance(palette[i])
		lumB := calculateLuminance(palette[j])
		return lumA > lumB
	})

	for len(palette) < numColorsTarget {
		log.Printf("Padding palette with black as it has %d colors, expected %d", len(palette), numColorsTarget)
		palette = append(palette, RGB{0, 0, 0})
	}
	if len(palette) > numColorsTarget {
		palette = palette[:numColorsTarget]
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

func calculateBitsPerPixelInternal(numColors int) (int, error) {
	if numColors <= 0 {
		return 0, fmt.Errorf("number of palette colors must be positive, got %d", numColors)
	}

	bpp := 0
	if numColors > 0 { 
		bpp = int(math.Ceil(math.Log2(float64(numColors))))
	}

	if bpp == 0 && numColors > 1 {
		bpp = 1
	} else if numColors == 1 {
		bpp = 1
	}

	if bpp == 0 {
		return 0, fmt.Errorf("calculated bitsPerPixel is 0 for %d palette colors, which is unexpected", numColors)
	}
	return bpp, nil
}

func compressImageInternal(inputPath string) ([]byte, []RGB, CompressionStats, error) {
	log.Printf("Starting compression of %s...", inputPath)

	inputFile, err := os.Open(inputPath)
	if err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to open input file: %w", err)
	}
	defer inputFile.Close()

	inputInfo, err := inputFile.Stat()
	if err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to get input file info: %w", err)
	}
	initialSize := inputInfo.Size()
	log.Printf("Initial size: %d bytes", initialSize)

	img, _, err := image.Decode(inputFile)
	if err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to decode image: %w", err)
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

	bitsPerPixel, err := calculateBitsPerPixelInternal(NumPaletteColors)
	if err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to calculate bits per pixel for encoding: %w", err)
	}
	log.Printf("Using %d bits per pixel for encoding.", bitsPerPixel)

	var paletteIndexedData bytes.Buffer
	var bitBuffer uint32
	var bitsInBuffer uint8

	bounds := rgbaImg.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			rVal, gVal, bVal, _ := rgbaImg.At(x, y).RGBA()
			pixelColor := RGB{uint8(rVal >> 8), uint8(gVal >> 8), uint8(bVal >> 8)}
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
	log.Printf("Size of palette-indexed pixel data (before zlib): %d bytes", sizeBeforeZlib)

	var zlibCompressedPixelData bytes.Buffer
	zlibWriter, _ := zlib.NewWriterLevel(&zlibCompressedPixelData, zlib.BestCompression)
	if _, err := zlibWriter.Write(paletteIndexedData.Bytes()); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("zlib compression of pixel data failed: %w", err)
	}
	if err := zlibWriter.Close(); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to close zlib writer for pixel data: %w", err)
	}
	log.Printf("Size of zlib-compressed pixel data: %d bytes", zlibCompressedPixelData.Len())

	var finalOutputData bytes.Buffer
	if err := binary.Write(&finalOutputData, binary.BigEndian, uint16(TargetWidth)); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to write TargetWidth: %w", err)
	}
	if err := binary.Write(&finalOutputData, binary.BigEndian, uint16(TargetHeight)); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to write TargetHeight: %w", err)
	}
	numStoredPaletteColors := uint8(len(dynamicPalette))
	if err := binary.Write(&finalOutputData, binary.BigEndian, numStoredPaletteColors); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to write number of palette colors: %w", err)
	}
	for _, pColor := range dynamicPalette {
		if _, err := finalOutputData.Write([]byte{pColor.R, pColor.G, pColor.B}); err != nil {
			return nil, nil, CompressionStats{}, fmt.Errorf("failed to write palette color: %w", err)
		}
	}
	if _, err := finalOutputData.Write(zlibCompressedPixelData.Bytes()); err != nil {
		return nil, nil, CompressionStats{}, fmt.Errorf("failed to append zlib data to final output: %w", err)
	}

	finalCompressedSize := int64(finalOutputData.Len())
	log.Printf("Total compressed output size (metadata + zlib pixel data): %d bytes", finalCompressedSize)

	var compressionRatio, spaceSaved float64
	if initialSize > 0 && finalCompressedSize > 0 {
		compressionRatio = float64(initialSize) / float64(finalCompressedSize)
		spaceSaved = (1.0 - (float64(finalCompressedSize) / float64(initialSize))) * 100.0
	}

	stats := CompressionStats{
		InitialSizeBytes:         initialSize,
		SizeBeforeZlibBytes:    sizeBeforeZlib,
		FinalCompressedSizeBytes: finalCompressedSize,
		CompressionRatio:         compressionRatio,
		SpaceSavedPercentage:     spaceSaved,
	}
	return finalOutputData.Bytes(), dynamicPalette, stats, nil
}

func decompressImageInternal(compressedData []byte, outputPath string) error {
	log.Printf("Starting color decompression (output: %s)...", outputPath)

	reader := bytes.NewReader(compressedData)

	var targetWRead, targetHRead uint16
	var numPaletteColorsRead uint8

	if err := binary.Read(reader, binary.BigEndian, &targetWRead); err != nil {
		return fmt.Errorf("failed to read targetWidth from metadata: %w", err)
	}
	if err := binary.Read(reader, binary.BigEndian, &targetHRead); err != nil {
		return fmt.Errorf("failed to read targetHeight from metadata: %w", err)
	}
	if err := binary.Read(reader, binary.BigEndian, &numPaletteColorsRead); err != nil {
		return fmt.Errorf("failed to read numPaletteColors from metadata: %w", err)
	}
	log.Printf("Read metadata: Target %dx%d, Palette Colors: %d", targetWRead, targetHRead, numPaletteColorsRead)

	readPalette := make([]RGB, numPaletteColorsRead)
	for i := 0; i < int(numPaletteColorsRead); i++ {
		rgbBytes := make([]byte, 3)
		if _, err := io.ReadFull(reader, rgbBytes); err != nil {
			return fmt.Errorf("failed to read palette color %d from metadata: %w", i, err)
		}
		readPalette[i] = RGB{rgbBytes[0], rgbBytes[1], rgbBytes[2]}
	}
	log.Printf("Read palette (%d colors) from metadata.", len(readPalette))

	zlibReader, err := zlib.NewReader(reader)
	if err != nil {
		return fmt.Errorf("failed to create zlib reader for pixel data: %w", err)
	}
	defer zlibReader.Close()

	paletteIndexedData, err := io.ReadAll(zlibReader)
	if err != nil {
		return fmt.Errorf("failed to decompress zlib pixel data: %w", err)
	}
	log.Printf("Size after zlib decompression (palette-indexed pixel data): %d bytes", len(paletteIndexedData))

	bitsPerPixel, err := calculateBitsPerPixelInternal(int(numPaletteColorsRead))
	if err != nil {
		return fmt.Errorf("failed to calculate bits per pixel for decompression: %w", err)
	}
	log.Printf("Using %d bits per pixel for decompression (from read numPaletteColors).", bitsPerPixel)

	decompressedImg := image.NewRGBA(image.Rect(0, 0, int(targetWRead), int(targetHRead)))
	var bitBuffer uint32
	var bitsInBuffer uint8
	byteIdx := 0
	totalPixels := int(targetWRead * targetHRead)
	pixelsDecoded := 0
	pixelMask := uint32((1 << bitsPerPixel) - 1)

	for y := 0; y < int(targetHRead); y++ {
		for x := 0; x < int(targetWRead); x++ {
			if pixelsDecoded >= totalPixels {
				break
			}

			for bitsInBuffer < uint8(bitsPerPixel) {
				if byteIdx < len(paletteIndexedData) {
					bitBuffer = (bitBuffer << 8) | uint32(paletteIndexedData[byteIdx])
					bitsInBuffer += 8
					byteIdx++
				} else {
					if bitsInBuffer == 0 && pixelsDecoded < totalPixels {
						return fmt.Errorf("unexpected end of compressed data. Decoded %d of %d pixels. byteIdx: %d, len_data: %d, bitsInBuf: %d, bpp: %d", pixelsDecoded, totalPixels, byteIdx, len(paletteIndexedData), bitsInBuffer, bitsPerPixel)
					}
					if pixelsDecoded < totalPixels {
						log.Printf("Warning: Potential end of stream or padding. bitsInBuf: %d, bpp: %d. At pixel %d/%d", bitsInBuffer, bitsPerPixel, pixelsDecoded, totalPixels)
					}
					goto endPixelLoop
				}
			}
		endPixelLoop:

			if bitsInBuffer >= uint8(bitsPerPixel) {
				shiftAmount := bitsInBuffer - uint8(bitsPerPixel)
				paletteIndex := uint8((bitBuffer >> shiftAmount) & pixelMask)

				if int(paletteIndex) >= len(readPalette) {
					return fmt.Errorf("decoded palette index %d out of bounds for palette size %d", paletteIndex, len(readPalette))
				}

				pColor := readPalette[paletteIndex]
				decompressedImg.Set(x, y, color.RGBA{pColor.R, pColor.G, pColor.B, 255})
				pixelsDecoded++

				bitsInBuffer -= uint8(bitsPerPixel)
				if bitsInBuffer > 0 {
					bitBuffer &= (1 << bitsInBuffer) - 1
				} else {
					bitBuffer = 0
				}
			} else if pixelsDecoded < totalPixels {
				log.Printf("Warning: Insufficient bits in buffer for pixel %d of %d. Needed: %d, Have: %d. Data might be truncated.", pixelsDecoded, totalPixels, bitsPerPixel, bitsInBuffer)
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
	log.Printf("Color decompressed image saved to %s", outputPath)
	return nil
}

func decompressImageGrayscaleInternal(compressedData []byte, outputPath string) error {
	log.Printf("Starting Grayscale decompression (output: %s)...", outputPath)

	reader := bytes.NewReader(compressedData)
	var targetWRead, targetHRead uint16
	var numPaletteColorsRead uint8

	if err := binary.Read(reader, binary.BigEndian, &targetWRead); err != nil {
		return fmt.Errorf("failed to read targetWidth (grayscale): %w", err)
	}
	if err := binary.Read(reader, binary.BigEndian, &targetHRead); err != nil {
		return fmt.Errorf("failed to read targetHeight (grayscale): %w", err)
	}
	if err := binary.Read(reader, binary.BigEndian, &numPaletteColorsRead); err != nil {
		return fmt.Errorf("failed to read numPaletteColors (grayscale): %w", err)
	}
	log.Printf("Read metadata (grayscale): Target %dx%d, Palette Colors: %d", targetWRead, targetHRead, numPaletteColorsRead)

	paletteByteSize := int64(numPaletteColorsRead) * 3
	if _, err := reader.Seek(paletteByteSize, io.SeekCurrent); err != nil {
		return fmt.Errorf("failed to seek past palette data (grayscale): %w", err)
	}

	zlibReader, err := zlib.NewReader(reader)
	if err != nil {
		return fmt.Errorf("failed to create zlib reader (grayscale): %w", err)
	}
	defer zlibReader.Close()

	paletteIndexedData, err := io.ReadAll(zlibReader)
	if err != nil {
		return fmt.Errorf("failed to decompress zlib data (grayscale): %w", err)
	}
	log.Printf("Size after zlib decompression (grayscale, pixel data): %d bytes", len(paletteIndexedData))

	bitsPerPixel, err := calculateBitsPerPixelInternal(int(numPaletteColorsRead))
	if err != nil {
		return fmt.Errorf("failed to calculate bits per pixel for grayscale decompression: %w", err)
	}
	log.Printf("Using %d bits per pixel for grayscale decompression.", bitsPerPixel)

	decompressedImg := image.NewGray(image.Rect(0, 0, int(targetWRead), int(targetHRead)))
	var bitBuffer uint32
	var bitsInBuffer uint8
	byteIdx := 0
	totalPixels := int(targetWRead * targetHRead)
	pixelsDecoded := 0
	pixelMask := uint32((1 << bitsPerPixel) - 1)

	for y := 0; y < int(targetHRead); y++ {
		for x := 0; x < int(targetWRead); x++ {
			if pixelsDecoded >= totalPixels {
				break
			}

			for bitsInBuffer < uint8(bitsPerPixel) {
				if byteIdx < len(paletteIndexedData) {
					bitBuffer = (bitBuffer << 8) | uint32(paletteIndexedData[byteIdx])
					bitsInBuffer += 8
					byteIdx++
				} else {
					if bitsInBuffer == 0 && pixelsDecoded < totalPixels {
						return fmt.Errorf("(Grayscale) Unexpected end of compressed data. Decoded %d of %d. byteIdx: %d, len: %d, bitsInBuf: %d, bpp: %d", pixelsDecoded, totalPixels, byteIdx, len(paletteIndexedData), bitsInBuffer, bitsPerPixel)
					}
					if pixelsDecoded < totalPixels {
						log.Printf("Warning (Grayscale): Potential end of stream/padding. bitsInBuf: %d, bpp: %d. Pixel %d/%d", bitsInBuffer, bitsPerPixel, pixelsDecoded, totalPixels)
					}
					goto endPixelLoopGray
				}
			}
		endPixelLoopGray:

			if bitsInBuffer >= uint8(bitsPerPixel) {
				shiftAmount := bitsInBuffer - uint8(bitsPerPixel)
				paletteIndex := uint8((bitBuffer >> shiftAmount) & pixelMask)

				var grayscaleValue uint8
				if numPaletteColorsRead <= 1 {
					if paletteIndex == 0 {
						grayscaleValue = 0
					} else {
						log.Printf("Warning (Grayscale): Palette index %d encountered with only %d palette colors defined. Using fallback grayscale value.", paletteIndex, numPaletteColorsRead)
						grayscaleValue = 128
					}
				} else {
					grayscaleValue = 255 - uint8(math.Round(float64(paletteIndex)*255.0/float64(int(numPaletteColorsRead)-1)))
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
				log.Printf("Warning (Grayscale): Insufficient bits for pixel %d of %d. Needed: %d, Have: %d.", pixelsDecoded, totalPixels, bitsPerPixel, bitsInBuffer)
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
		log.Printf("Sample input image %s already exists.", filePath)
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

	encodeCmd := flag.NewFlagSet("encode", flag.ExitOnError)
	encodeInput := encodeCmd.String("input", "input.jpg", "Input image path")
	encodeOutput := encodeCmd.String("output", "outputs_go/compressed.bin", "Output compressed file path")

	decodeCmd := flag.NewFlagSet("decode", flag.ExitOnError)
	decodeInput := decodeCmd.String("input", "outputs_go/compressed.bin", "Input compressed file path")
	decodeOutput := decodeCmd.String("output", "outputs_go/decompressed.png", "Output decompressed color image path")
	decodeGrayscaleOutput := decodeCmd.String("grayscale", "", "Optional: Output decompressed grayscale image path (e.g., outputs_go/decompressed_gray.png)")

	if len(os.Args) < 2 {
		fmt.Println("Expected 'encode' or 'decode' subcommand.")
		fmt.Println("\nUsage for encode:")
		encodeCmd.PrintDefaults()
		fmt.Println("\nUsage for decode:")
		decodeCmd.PrintDefaults()
		os.Exit(1)
	}

	_ = os.Mkdir("outputs_go", 0755)

	switch os.Args[1] {
	case "encode":
		encodeCmd.Parse(os.Args[2:])
		if *encodeInput == "" || *encodeOutput == "" {
			encodeCmd.Usage()
			os.Exit(1)
		}

		if *encodeInput == "input.jpg" {
			if err := createSampleInputImage(*encodeInput); err != nil {
				log.Fatalf("Failed to ensure sample input image %s: %v", *encodeInput, err)
			}
		}

		log.Printf("Encoding %s to %s...", *encodeInput, *encodeOutput)
		compressedBytes, _, stats, err := compressImageInternal(*encodeInput)
		if err != nil {
			log.Fatalf("Compression failed: %v", err)
		}

		err = os.WriteFile(*encodeOutput, compressedBytes, 0644)
		if err != nil {
			log.Fatalf("Failed to write compressed file %s: %v", *encodeOutput, err)
		}

		fmt.Println("\n--- Go Compression Report ---")
		fmt.Printf("Input image: %s (%d bytes)\n", *encodeInput, stats.InitialSizeBytes)
		fmt.Printf("Output compressed file: %s (%d bytes)\n", *encodeOutput, stats.FinalCompressedSizeBytes)
		fmt.Printf("Size of pixel data before zlib: %d bytes\n", stats.SizeBeforeZlibBytes)
		if stats.FinalCompressedSizeBytes > 0 && stats.InitialSizeBytes > 0 {
			fmt.Printf("Overall compression ratio: %.2f:1\n", stats.CompressionRatio)
			fmt.Printf("Overall space saved: %.2f%%\n", stats.SpaceSavedPercentage)
		} else {
			fmt.Println("Overall compression ratio: N/A (zero size involved)")
			fmt.Println("Overall space saved: N/A")
		}
		fmt.Printf("Encoder settings used:\n")
		fmt.Printf("  Target dimensions: %dx%d\n", TargetWidth, TargetHeight)
		bppEncode, _ := calculateBitsPerPixelInternal(NumPaletteColors)
		fmt.Printf("  Palette colors: %d (%d bpp for pixel data)\n", NumPaletteColors, bppEncode)
		fmt.Printf("  Quantization shift: %d\n", QuantizationShiftAmount)
		fmt.Printf("  Blur sigma: %.2f\n", BlurSigma)
		fmt.Println("--- End of Report ---")
		log.Println("Encoding process completed.")

	case "decode":
		decodeCmd.Parse(os.Args[2:])
		if *decodeInput == "" || *decodeOutput == "" {
			decodeCmd.Usage()
			os.Exit(1)
		}

		log.Printf("Decoding %s to %s...", *decodeInput, *decodeOutput)
		compressedBytes, err := os.ReadFile(*decodeInput)
		if err != nil {
			log.Fatalf("Failed to read compressed file %s: %v", *decodeInput, err)
		}
		log.Printf("Read %d bytes from compressed file %s.", len(compressedBytes), *decodeInput)

		err = decompressImageInternal(compressedBytes, *decodeOutput)
		if err != nil {
			log.Fatalf("Color decompression failed: %v", err)
		}

		if *decodeGrayscaleOutput != "" {
			log.Printf("Also decoding %s to %s (grayscale)...", *decodeInput, *decodeGrayscaleOutput)
			err = decompressImageGrayscaleInternal(compressedBytes, *decodeGrayscaleOutput)
			if err != nil {
				log.Fatalf("Grayscale decompression failed: %v", err)
			}
		}
		log.Println("Decoding process completed.")

	default:
		fmt.Printf("Unknown subcommand: %s\n", os.Args[1])
		fmt.Println("Expected 'encode' or 'decode'.")
		os.Exit(1)
	}
}