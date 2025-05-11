package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"image/png"
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

func findClosestPaletteColorIndex(c color.Color, palette []color.RGBA) int {
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
	return minIndex
}

func Encode(s string) string {
	if len(s) == 0 {
		return ""
	}

	var encoded strings.Builder
	i := 0
	for i < len(s) {
		char := s[i]
		count := 0
		j := i
		for j < len(s) && s[j] == char {
			count++
			j++
		}

		encoded.WriteString("!")
		encoded.WriteByte(char)
		encoded.WriteString(strconv.Itoa(count))
		i = j
	}
	return encoded.String()
}

func Decode(s string) string {
	if !strings.Contains(s, "!") {
		return s
	}

	groups := strings.Split(strings.TrimSpace(s), "!")

	var decoded strings.Builder
	for _, group := range groups {
		if group == "" {
			continue
		}

		dataChar := group[0]
		lengthStr := group[1:]
		length, err := strconv.Atoi(lengthStr)
		if err != nil {
			panic(fmt.Sprintf("Decode error: could not parse length '%s' from group '%s': %v", lengthStr, group, err))
		}

		for k := 0; k < length; k++ {
			decoded.WriteByte(dataChar)
		}
	}
	return decoded.String()
}

func writeToFile(filePath, content string) error {
	outputFile, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("could not create file '%s': %v", filePath, err)
	}
	defer outputFile.Close()

	writer := bufio.NewWriter(outputFile)
	_, err = writer.WriteString(content)
	if err != nil {
		return fmt.Errorf("could not write content to file '%s': %v", filePath, err)
	}
	err = writer.Flush()
	if err != nil {
		return fmt.Errorf("could not flush data to file '%s': %v", filePath, err)
	}
	return nil
}

func processImage(imagePath, uncompressedOutputPath string, targetHeight int) {
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

	if originalHeight == 0 {
		fmt.Println("error: image original height is zero.")
		os.Exit(1)
	}

	aspectRatio := float64(originalWidth) / float64(originalHeight)
	targetWidth := int(float64(targetHeight) * aspectRatio)

	if targetWidth < 1 {
		fmt.Printf("warning: calculated target width is %d. Setting to 1.\n", targetWidth)
		targetWidth = 1
	}

	resizedImg := imaging.Resize(img, targetWidth, targetHeight, imaging.Lanczos)

	var lines []string
	for y := 0; y < targetHeight; y++ {
		var lineChars strings.Builder
		for x := 0; x < targetWidth; x++ {
			pixelColor := resizedImg.At(x, y)
			paletteIndex := findClosestPaletteColorIndex(pixelColor, targetPalette)
			lineChars.WriteString(strconv.Itoa(paletteIndex))
		}
		lines = append(lines, lineChars.String())
	}

	fullUncompressedText := strings.Join(lines, "\n")

	err = writeToFile(uncompressedOutputPath, fullUncompressedText)
	if err != nil {
		fmt.Printf("error writing uncompressed file: %v\n", err)
		os.Exit(1)
	}

	var encodedImage string
	encodedImage = Encode(strings.Join(lines, ""))

	imageWithHeaders := fmt.Sprintf("%d|%s", targetWidth, encodedImage)

	ext := filepath.Ext(uncompressedOutputPath)
	baseName := strings.TrimSuffix(uncompressedOutputPath, ext)
	compressedOutputPath := baseName + "_compressed" + ext

	err = writeToFile(compressedOutputPath, imageWithHeaders)
	if err != nil {
		fmt.Printf("error writing compressed file: %v\n", err)
		os.Exit(1)
	}

	uncompressedBytes := len(fullUncompressedText)
	compressedBytes := len(imageWithHeaders)

	fmt.Printf("\n--- Compression Report ---\n")
	if uncompressedBytes == 0 && compressedBytes == 0 && len(lines) == 0 {
		fmt.Println("  Input resulted in empty text, no compression data.")
	} else if compressedBytes == 0 && uncompressedBytes > 0 {
		fmt.Printf("  Uncompressed Bytes: %d\n", uncompressedBytes)
		fmt.Printf("  Compressed Bytes: %d (Compression resulted in empty output)\n", compressedBytes)
		fmt.Println("  Compression Ratio: Undefined (compressed to zero bytes)")
	} else if uncompressedBytes == 0 {
		fmt.Printf("  Uncompressed Bytes: %d\n", uncompressedBytes)
		fmt.Printf("  Compressed Bytes: %d\n", compressedBytes)
		fmt.Println("  Compression Ratio: Undefined (uncompressed is zero bytes)")
	} else {
		directRatio := float64(uncompressedBytes) / float64(compressedBytes)
		pyStyleRatioPercent := (float64(compressedBytes) / float64(uncompressedBytes)) * 100.0

		fmt.Printf("  Uncompressed Bytes: %d\n", uncompressedBytes)
		fmt.Printf("  Compressed Bytes: %d\n", compressedBytes)
		fmt.Printf("  Direct Ratio (Uncompressed/Compressed): %.2f\n", directRatio)
		fmt.Printf("  Size Percentage (Compressed/Uncompressed * 100): %.2f%%\n", pyStyleRatioPercent)
	}
	fmt.Println("--- End Compression Report ---\n")
}

func reconstructImage(decompressedTextLines []string, outputPath string) error {
	if len(decompressedTextLines) == 0 {
		return fmt.Errorf("no lines to decompress into an image")
	}

	height := len(decompressedTextLines)
	width := 0
	firstNonEmptyLineIdx := -1

	for i, line := range decompressedTextLines {
		if len(line) > 0 {
			width = len(line)
			firstNonEmptyLineIdx = i
			break
		}
	}

	if firstNonEmptyLineIdx == -1 {
		fmt.Printf("warning: All decompressed lines are empty. Outputting a 1x%d image using the first palette color.\n", height)
		width = 1
		img := image.NewRGBA(image.Rect(0, 0, width, height))

		defaultColor := color.RGBA{R: 0, G: 0, B: 0, A: 255}
		if len(targetPalette) > 0 {
			defaultColor = targetPalette[0]
		}

		for y := 0; y < height; y++ {
			img.SetRGBA(0, y, defaultColor)
		}
		outFile, err := os.Create(outputPath)
		if err != nil {
			return fmt.Errorf("could not create output image file '%s': %v", outputPath, err)
		}
		defer outFile.Close()
		if err := png.Encode(outFile, img); err != nil {
			return fmt.Errorf("could not encode (empty) image to PNG '%s': %v", outputPath, err)
		}
		fmt.Printf("successfully wrote minimal 1x%d image with first palette color: '%s'\n", height, outputPath)
		return nil
	}

	for i, line := range decompressedTextLines {
		if len(line) > 0 && len(line) != width {
			preview := line
			if len(preview) > 15 {
				preview = preview[:15] + "..."
			}
			return fmt.Errorf("inconsistent line lengths in decompressed text: line %d ('%s') has length %d, expected %d (from line %d)", i+1, preview, len(line), width, firstNonEmptyLineIdx+1)
		}
	}

	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y, lineStr := range decompressedTextLines {
		if len(lineStr) == 0 {
			defaultColor := color.RGBA{R: 0, G: 0, B: 0, A: 255}
			if len(targetPalette) > 0 {
				defaultColor = targetPalette[0]
			}
			for x := 0; x < width; x++ {
				img.SetRGBA(x, y, defaultColor)
			}
			continue
		}

		for x, charRune := range lineStr {
			paletteIndex, err := strconv.Atoi(string(charRune))
			if err != nil {
				return fmt.Errorf("invalid character '%c' in decompressed text (not a digit for palette index) at line %d, char %d: %v", charRune, y+1, x+1, err)
			}
			if paletteIndex < 0 || paletteIndex >= len(targetPalette) {
				return fmt.Errorf("palette index '%c' (parsed as %d) is out of range (0-%d) for target palette at line %d, char %d", charRune, paletteIndex, len(targetPalette)-1, y+1, x+1)
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
	fmt.Printf("successfully decompressed text to image: '%s'\n", outputPath)
	return nil
}

func main() {
	//imagePath := "/home/radi8/Downloads/Dropped Image.png"
	imagePath := "/home/radi8/Downloads/PXL_20240723_044204518.jpg"
	uncompressedBase := "image_palettized.txt"
	targetHeight := 128

	processImage(imagePath, uncompressedBase, targetHeight)

	ext := filepath.Ext(uncompressedBase)
	baseName := strings.TrimSuffix(uncompressedBase, ext)
	compressedFilePath := baseName + "_compressed" + ext
	reconstructedImagePath := baseName + "_reconstructed.png"

	file, err := os.Open(compressedFilePath)
	if err != nil {
		fmt.Printf("Error: Could not open compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		fmt.Printf("Error: Could not get file info for '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}
	fileSize := fileInfo.Size()
	buffer := make([]byte, fileSize)
	_, err = file.Read(buffer)
	if err != nil {
		fmt.Printf("Error: Could not read compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}

	parts := strings.SplitN(string(buffer), "|", 2)
	if len(parts) != 2 {
		fmt.Printf("Error: Invalid compressed file format in '%s'. Expected 'width|encoded_data'.\n", compressedFilePath)
		os.Exit(1)
	}

	imgWidth, err := strconv.Atoi(parts[0])
	if err != nil {
		fmt.Printf("Error: Invalid width format in compressed file '%s': %v\n", compressedFilePath, err)
		os.Exit(1)
	}

	encodedData := parts[1]
	decodedData := Decode(encodedData)

	var decompressedLines []string
	for i := 0; i < len(decodedData); i += imgWidth {
		end := i + imgWidth
		if end > len(decodedData) {
			end = len(decodedData)
		}

		decompressedLines = append(decompressedLines, decodedData[i:end])
	}

	err = reconstructImage(decompressedLines, reconstructedImagePath)
	if err != nil {
		fmt.Printf("Error converting decompressed text to image: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nProcessing complete.\nUncompressed (palettized) text: %s\nCompressed text: %s\nReconstructed image: %s\n",
		uncompressedBase, compressedFilePath, reconstructedImagePath)
}
