package main

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/disintegration/imaging"
)

var targetPalette = []color.RGBA{
	{R: 0, G: 0, B: 0, A: 255},       // 0: #000000
	{R: 156, G: 103, B: 54, A: 255},  // 1: #9c6736
	{R: 10, G: 53, B: 80, A: 255},    // 2: #0a3550
	{R: 0, G: 12, B: 54, A: 255},     // 3: #000c36
	{R: 63, G: 87, B: 173, A: 255},   // 4: #3f57ad
	{R: 15, G: 66, B: 168, A: 255},   // 5: #0f42a8
	{R: 91, G: 157, B: 215, A: 255},  // 6: #5b9dd7
	{R: 3, G: 6, B: 52, A: 255},      // 7: #030634
	{R: 252, G: 255, B: 255, A: 255}, // 8: #fcffff
	{R: 0, G: 49, B: 116, A: 255},    // 9: #003174
}

func findClosestPaletteColorIndex(c color.Color, palette []color.RGBA) byte {
	r, g, b, _ := c.RGBA()
	r8 := uint8(r >> 8)
	g8 := uint8(g >> 8)
	b8 := uint8(b >> 8)
	minIndex := 0
	minDistSq := int64(math.MaxInt64)
	for i, pColor := range palette {
		dr := int64(r8) - int64(pColor.R)
		dg := int64(g8) - int64(pColor.G)
		db := int64(b8) - int64(pColor.B)
		distSq := dr*dr + dg*dg + db*db
		if distSq < minDistSq {
			minDistSq = distSq
			minIndex = i
		}
		if distSq == 0 {
			break
		}
	}
	return byte(minIndex)
}

func packPixelData(indices []byte) []byte {
	numIndices := len(indices)
	if numIndices == 0 {
		return []byte{}
	}
	packedLen := (numIndices + 1) / 2
	packed := make([]byte, packedLen)
	for i := 0; i < numIndices; i += 2 {
		idx1 := indices[i] & 0x0F
		var idx2 byte = 0
		if i+1 < numIndices {
			idx2 = indices[i+1] & 0x0F
		}
		packed[i/2] = (idx1 << 4) | idx2
	}
	return packed
}

func unpackPixelData(packed []byte, totalPixels int) ([]byte, error) {
	if totalPixels == 0 {
		return []byte{}, nil
	}
	expectedMinPackedLen := (totalPixels + 1) / 2
	if len(packed) < expectedMinPackedLen {
		return nil, fmt.Errorf("not enough data in packed bytes: expected at least %d bytes for %d pixels, got %d bytes", expectedMinPackedLen, totalPixels, len(packed))
	}
	indices := make([]byte, 0, totalPixels)
	for i := 0; i < expectedMinPackedLen; i++ {
		b := packed[i]
		idx1 := b >> 4
		idx2 := b & 0x0F
		if len(indices) < totalPixels {
			indices = append(indices, idx1)
		} else {
			break
		}
		if len(indices) < totalPixels {
			indices = append(indices, idx2)
		} else {
			break
		}
	}
	if len(indices) != totalPixels {
		return nil, fmt.Errorf("failed to unpack the correct number of pixels: expected %d, got %d. Input packed len: %d, expectedMinPackedLen: %d", totalPixels, len(indices), len(packed), expectedMinPackedLen)
	}
	return indices, nil
}

func compressWithZlib(data []byte) ([]byte, error) {
	var b bytes.Buffer
	w, err := zlib.NewWriterLevel(&b, zlib.BestCompression)
	if err != nil {
		return nil, fmt.Errorf("zlib NewWriterLevel error: %w", err)
	}
	_, err = w.Write(data)
	if err != nil {
		w.Close()
		return nil, fmt.Errorf("zlib write error: %w", err)
	}
	err = w.Close()
	if err != nil {
		return nil, fmt.Errorf("zlib close error: %w", err)
	}
	return b.Bytes(), nil
}

func decompressWithZlib(compressedData []byte) ([]byte, error) {
	if len(compressedData) == 0 {
		return []byte{}, nil
	}
	b := bytes.NewReader(compressedData)
	r, err := zlib.NewReader(b)
	if err != nil {
		return nil, fmt.Errorf("zlib NewReader error: %w", err)
	}
	defer r.Close()
	decompressedData, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("zlib ReadAll error: %w", err)
	}
	return decompressedData, nil
}

func writeStringToFile(filePath, content string) error {
	err := os.WriteFile(filePath, []byte(content), 0644)
	if err != nil {
		return fmt.Errorf("could not write content to file '%s': %v", filePath, err)
	}
	return nil
}


func bytesToBitString(data []byte) string {
	if len(data) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.Grow(len(data) * 8)
	for _, b := range data {
		for i := 7; i >= 0; i-- {
			if (b>>i)&1 == 1 {
				sb.WriteByte('1')
			} else {
				sb.WriteByte('0')
			}
		}
	}
	return sb.String()
}

func processImage(imagePath, uncompressedOutputPath string, targetHeight int) (processedWidth int, processedHeight int) {
	file, err := os.Open(imagePath)
	if err != nil {
		fmt.Printf("error: could not open image file '%s': %v\n", imagePath, err)
		os.Exit(1)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		fmt.Printf("error: could not decode image '%s': %v\n", imagePath, err)
		os.Exit(1)
	}

	bounds := img.Bounds()
	originalWidth := bounds.Dx()
	originalHeight := bounds.Dy()

	if originalHeight <= 0 {
		fmt.Printf("error: image original height (%d) is not positive.\n", originalHeight)
		os.Exit(1)
	}
	if targetHeight < 1 {
		fmt.Printf("warning: target height %d is less than 1. Setting to 1.\n", targetHeight)
		targetHeight = 1
	}

	aspectRatio := float64(originalWidth) / float64(originalHeight)
	targetWidth := int(math.Round(float64(targetHeight) * aspectRatio))

	if targetWidth < 1 {
		fmt.Printf("warning: calculated target width is %d. Setting to 1.\n", targetWidth)
		targetWidth = 1
	}
	processedWidth = targetWidth
	processedHeight = targetHeight

	resizedImg := imaging.Resize(img, targetWidth, targetHeight, imaging.Lanczos)

	allPaletteIndices := make([]byte, 0, targetWidth*targetHeight)
	var humanReadableLines []string
	for y := 0; y < targetHeight; y++ {
		var lineChars strings.Builder
		for x := 0; x < targetWidth; x++ {
			pixelColor := resizedImg.At(x, y)
			paletteIndex := findClosestPaletteColorIndex(pixelColor, targetPalette)
			allPaletteIndices = append(allPaletteIndices, paletteIndex)
			lineChars.WriteString(strconv.Itoa(int(paletteIndex)))
		}
		humanReadableLines = append(humanReadableLines, lineChars.String())
	}

	fullUncompressedText := strings.Join(humanReadableLines, "\n")
	err = writeStringToFile(uncompressedOutputPath, fullUncompressedText)
	if err != nil {
		fmt.Printf("error writing uncompressed palettized file: %v\n", err)
		os.Exit(1)
	}

	ext := filepath.Ext(uncompressedOutputPath)
	baseName := strings.TrimSuffix(uncompressedOutputPath, ext)

	packedData := packPixelData(allPaletteIndices)

	packedForZlibPath := baseName + "_packed_for_zlib.dat"
	errWritePackedDat := os.WriteFile(packedForZlibPath, packedData, 0644)
	if errWritePackedDat != nil {
		fmt.Printf("Warning: error writing packed data (binary) to '%s': %v\n", packedForZlibPath, errWritePackedDat)
	} else {
		fmt.Printf("Successfully wrote packed data (binary, pre-zlib) to: %s\n", packedForZlibPath)
	}


	packedBitsString := bytesToBitString(packedData)
	packedBitsPath := baseName + "_packed_bits.txt"
	errWritePackedBits := writeStringToFile(packedBitsPath, packedBitsString)
	if errWritePackedBits != nil {
		fmt.Printf("Warning: error writing packed data (ASCII bits) to '%s': %v\n", packedBitsPath, errWritePackedBits)
	} else {
		fmt.Printf("Successfully wrote packed data (ASCII bits) to: %s\n", packedBitsPath)
	}

	fmt.Println("--- Packed Data Bits (ASCII Preview, pre-zlib) ---")
	previewLen := len(packedBitsString)
	maxPreviewLen := 256
	if previewLen > maxPreviewLen {
		fmt.Printf("%s... (full string in %s)\n", packedBitsString[:maxPreviewLen], packedBitsPath)
	} else if previewLen > 0 {
		fmt.Println(packedBitsString)
	} else {
		fmt.Println("(empty)")
	}
	fmt.Println("--- End Packed Data Bits Preview ---")

	zlibCompressedData, err := compressWithZlib(packedData)
	if err != nil {
		fmt.Printf("error compressing data with zlib: %v\n", err)
		os.Exit(1)
	}

	compressedOutputPath := baseName + "_compressed" + ext
	compFile, err := os.Create(compressedOutputPath)
	if err != nil {
		fmt.Printf("error creating compressed file '%s': %v\n", compressedOutputPath, err)
		os.Exit(1)
	}
	header := fmt.Sprintf("%d\n%d\n", targetWidth, targetHeight)
	_, err = compFile.WriteString(header)
	if err != nil {
		compFile.Close()
		fmt.Printf("error writing header to compressed file: %v\n", err)
		os.Exit(1)
	}
	_, err = compFile.Write(zlibCompressedData)
	if err != nil {
		compFile.Close()
		fmt.Printf("error writing zlib data to compressed file: %v\n", err)
		os.Exit(1)
	}
	compFile.Close()

	uncompressedSizeForReport := targetWidth * targetHeight
	compressedFileInfo, err := os.Stat(compressedOutputPath)
	var compressedSizeForReport int64 = -1
	if err == nil {
		compressedSizeForReport = compressedFileInfo.Size()
	} else {
		fmt.Printf("warning: could not stat compressed file '%s' for size report: %v\n", compressedOutputPath, err)
	}

	fmt.Printf("\n--- Compression Report ---\n")
	fmt.Printf("  Image Dimensions (processed): %d x %d\n", targetWidth, targetHeight)
	fmt.Printf("  Uncompressed Palette Indices (1 char/pixel value in .txt): %d bytes\n", uncompressedSizeForReport)
	if errWritePackedDat == nil {
		fmt.Printf("  Raw Packed Data (binary, pre-zlib) saved to: %s\n", packedForZlibPath)
	}
	if errWritePackedBits == nil {
		fmt.Printf("  Raw Packed Data (ASCII bits, pre-zlib) saved to: %s\n", packedBitsPath)
	}
	fmt.Printf("  Size of Raw Packed Data (fed to zlib): %d bytes (%d bits)\n", len(packedData), len(packedData)*8)
	fmt.Printf("  Zlib compressed payload size: %d bytes\n", len(zlibCompressedData))
	if compressedSizeForReport != -1 {
		fmt.Printf("  Total Compressed File Size (header + zlib payload): %d bytes\n", compressedSizeForReport)
		if uncompressedSizeForReport > 0 && compressedSizeForReport > 0 {
			directRatio := float64(uncompressedSizeForReport) / float64(compressedSizeForReport)
			sizePercentage := (float64(compressedSizeForReport) / float64(uncompressedSizeForReport)) * 100.0
			fmt.Printf("  Direct Ratio (Palettized Text / Compressed File): %.2f\n", directRatio)
			fmt.Printf("  Size Percentage (Compressed File / Palettized Text * 100): %.2f%%\n", sizePercentage)
		} else if uncompressedSizeForReport == 0 {
			fmt.Println("  Compression Ratio: Undefined (uncompressed size is zero)")
		} else {
			fmt.Println("  Compression Ratio: Undefined (compressed file size is zero)")
		}
	} else {
		fmt.Println("  Total Compressed File Size: Unknown")
	}
	fmt.Println("--- End Compression Report ---")
	return
}

func reconstructImageFromIndices(indices []byte, width int, height int, outputPath string) error {
	if width <= 0 || height <= 0 {
		if width == 0 && height == 0 && len(indices) == 0 {
			fmt.Printf("Reconstructing a 0x0 image. Output will be an empty PNG: '%s'\n", outputPath)
			img := image.NewRGBA(image.Rect(0, 0, 0, 0))
			outFile, err := os.Create(outputPath)
			if err != nil {
				return fmt.Errorf("could not create output image file '%s': %v", outputPath, err)
			}
			defer outFile.Close()
			return png.Encode(outFile, img)
		}
		return fmt.Errorf("invalid dimensions for reconstruction: width=%d, height=%d. Indices count: %d", width, height, len(indices))
	}

	if len(indices) != width*height {
		return fmt.Errorf("mismatch between indices count (%d) and width*height (%d*%d=%d)", len(indices), width, height, width*height)
	}

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idxOffset := y*width + x
			paletteIndex := indices[idxOffset]
			if int(paletteIndex) < 0 || int(paletteIndex) >= len(targetPalette) {
				return fmt.Errorf("palette index %d at (%d,%d) is out of range (0-%d)", paletteIndex, x, y, len(targetPalette)-1)
			}
			pixelColor := targetPalette[paletteIndex]
			img.SetRGBA(x, y, pixelColor)
		}
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("could not create output image file '%s': %v", outputPath, err)
	}
	defer outFile.Close()

	if err := png.Encode(outFile, img); err != nil {
		return fmt.Errorf("could not encode image to PNG '%s': %v", outputPath, err)
	}
	fmt.Printf("Successfully reconstructed image from indices: '%s'\n", outputPath)
	return nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run yourprogram.go <image_path> [target_height]")
		fmt.Println("Example: go run yourprogram.go /path/to/image.png 32")
		os.Exit(1)
	}
	imagePath := os.Args[1]

	targetHeightInput := 32
	if len(os.Args) > 2 {
		parsedHeight, err := strconv.Atoi(os.Args[2])
		if err != nil {
			fmt.Printf("Invalid target height '%s', using default %d: %v\n", os.Args[2], targetHeightInput, err)
		} else {
			targetHeightInput = parsedHeight
		}
	}

	inputFileName := filepath.Base(imagePath)
	inputExt := filepath.Ext(inputFileName)
	inputBaseName := strings.TrimSuffix(inputFileName, inputExt)

	outputDir := "."

	uncompressedBaseFile := filepath.Join(outputDir, inputBaseName+"_palettized.txt")

	processedWidth, processedHeight := processImage(imagePath, uncompressedBaseFile, targetHeightInput)

	palettizedFileExt := filepath.Ext(uncompressedBaseFile)
	palettizedBaseName := strings.TrimSuffix(uncompressedBaseFile, palettizedFileExt)

	compressedFilePath := palettizedBaseName + "_compressed" + palettizedFileExt
	reconstructedImagePath := palettizedBaseName + "_reconstructed.png"
	packedForZlibPath := palettizedBaseName + "_packed_for_zlib.dat"
	packedBitsPath := palettizedBaseName + "_packed_bits.txt"

	compFile, err := os.Open(compressedFilePath)
	if err != nil {
		fmt.Printf("Error: Could not open compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}
	defer compFile.Close()

	reader := bufio.NewReader(compFile)
	widthStr, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error: Could not read width from compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}
	imgWidth, err := strconv.Atoi(strings.TrimSpace(widthStr))
	if err != nil {
		fmt.Printf("Error: Invalid width format in compressed file '%s' ('%s'): %v\n", compressedFilePath, widthStr, err)
		os.Exit(1)
	}

	heightStr, err := reader.ReadString('\n')
	if err != nil {
		fmt.Printf("Error: Could not read height from compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}
	imgHeight, err := strconv.Atoi(strings.TrimSpace(heightStr))
	if err != nil {
		fmt.Printf("Error: Invalid height format in compressed file '%s' ('%s'): %v\n", compressedFilePath, heightStr, err)
		os.Exit(1)
	}

	zlibCompressedData, err := io.ReadAll(reader)
	if err != nil {
		fmt.Printf("Error: Could not read zlib data from compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}

	packedPixelData, err := decompressWithZlib(zlibCompressedData)
	if err != nil {
		fmt.Printf("Error: Could not decompress zlib data: %v\n", err)
		os.Exit(1)
	}

	totalPixels := imgWidth * imgHeight
	allPaletteIndices, err := unpackPixelData(packedPixelData, totalPixels)
	if err != nil {
		fmt.Printf("Error: Could not unpack pixel data (expected dims: %dx%d, totalPixels: %d, packedLen: %d): %v\n", imgWidth, imgHeight, totalPixels, len(packedPixelData), err)
		os.Exit(1)
	}

	err = reconstructImageFromIndices(allPaletteIndices, imgWidth, imgHeight, reconstructedImagePath)
	if err != nil {
		fmt.Printf("Error reconstructing image: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n--- File Summary ---\n")
	fmt.Printf("  Input image: %s\n", imagePath)
	fmt.Printf("  Processed to dimensions: %d x %d (based on target height %d)\n", processedWidth, processedHeight, targetHeightInput)
	fmt.Printf("  Palettized text (human-readable indices): %s\n", uncompressedBaseFile)
	if _, errStat := os.Stat(packedForZlibPath); errStat == nil {
		fmt.Printf("  Raw Packed Data (binary, pre-zlib bytes): %s\n", packedForZlibPath)
	}
	if _, errStat := os.Stat(packedBitsPath); errStat == nil {
		fmt.Printf("  Raw Packed Data (ASCII bits, pre-zlib): %s\n", packedBitsPath)
	}
	fmt.Printf("  Compressed data file (header + zlib): %s\n", compressedFilePath)
	fmt.Printf("  Reconstructed image: %s\n", reconstructedImagePath)
	fmt.Println("--- End File Summary ---")
}