package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"bytes"
	"compress/zlib"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"
	"sort"
	"unsafe"

	"github.com/disintegration/gift"
	"github.com/nfnt/resize"
)

func init() {
	image.RegisterFormat("jpeg", "jpeg", jpeg.Decode, jpeg.DecodeConfig)
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

type RGB struct {
	R, G, B uint8
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

func getBitsPerPixel(numPaletteColors int) (int, error) {
	if numPaletteColors <= 0 {
		return 0, fmt.Errorf("NumPaletteColors must be > 0, got %d", numPaletteColors)
	}
	if numPaletteColors == 1 {
		return 1, nil
	}
	bpp := int(math.Ceil(math.Log2(float64(numPaletteColors))))
	if bpp == 0 {
		return 0, fmt.Errorf("calculated bitsPerPixel is 0 with %d palette colors", numPaletteColors)
	}
	return bpp, nil
}

func processAndPaletteIndex(img image.Image, targetWidth, targetHeight uint, numPaletteColors int, blurSigma float32, quantizationShiftAmount uint) (*image.RGBA, []RGB, []byte, error) {
	resizedImg := resize.Resize(targetWidth, targetHeight, img, resize.NearestNeighbor)
	g := gift.New(gift.GaussianBlur(blurSigma))
	blurredImg := image.NewRGBA(g.Bounds(resizedImg.Bounds()))
	g.Draw(blurredImg, resizedImg)

	rgbaImg := image.NewRGBA(blurredImg.Bounds())
	draw.Draw(rgbaImg, rgbaImg.Bounds(), blurredImg, image.Point{}, draw.Src)

	dynamicPalette := generateDynamicPalette(rgbaImg, numPaletteColors, quantizationShiftAmount)

	bitsPerPixel, err := getBitsPerPixel(numPaletteColors)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("error calculating bits per pixel: %w", err)
	}

	var paletteIndexedData bytes.Buffer
	var bitBuffer uint32
	var bitsInBuffer uint8

	bounds := rgbaImg.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g_val, b, _ := rgbaImg.At(x, y).RGBA()
			pixelColor := RGB{uint8(r >> 8), uint8(g_val >> 8), uint8(b >> 8)}
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
	return rgbaImg, dynamicPalette, paletteIndexedData.Bytes(), nil
}

//export FreeMem
func FreeMem(ptr unsafe.Pointer) {
	C.free(ptr)
}


//export ProcessImageToOutputJPEG
func ProcessImageToOutputJPEG(
	inputImageBytes unsafe.Pointer, inputImageLen C.int,
	targetWidth C.uint, targetHeight C.uint,
	numPaletteColors C.int, blurSigma C.float, quantizationShiftAmount C.uint,
	outputJPEG **C.char, outputJPEGLen *C.int,
	outputPaletteJson **C.char, outputPaletteJsonLen *C.int,
) *C.char {
	goInputBytes := C.GoBytes(inputImageBytes, inputImageLen)

	img, _, err := image.Decode(bytes.NewReader(goInputBytes))
	if err != nil {
		return C.CString(fmt.Sprintf("failed to decode input image: %v", err))
	}

	resizedImg := resize.Resize(uint(targetWidth), uint(targetHeight), img, resize.NearestNeighbor)

	g := gift.New(gift.GaussianBlur(float32(blurSigma)))
	blurredImg := image.NewRGBA(g.Bounds(resizedImg.Bounds()))
	g.Draw(blurredImg, resizedImg)

	rgbaImg := image.NewRGBA(blurredImg.Bounds())
	draw.Draw(rgbaImg, rgbaImg.Bounds(), blurredImg, image.Point{}, draw.Src)

	dynamicPalette := generateDynamicPalette(rgbaImg, int(numPaletteColors), uint(quantizationShiftAmount))
	paletteJSONBytes, err := json.Marshal(dynamicPalette)
	if err != nil {
		return C.CString(fmt.Sprintf("failed to marshal palette to JSON: %v", err))
	}

	palettizedViewImg := image.NewRGBA(rgbaImg.Bounds())
	bounds := rgbaImg.Bounds()
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r_val, g_val, b_val, a_val := rgbaImg.At(x, y).RGBA()
			pixelColor := RGB{uint8(r_val >> 8), uint8(g_val >> 8), uint8(b_val >> 8)}
			paletteIndex := findClosestPaletteIndex(pixelColor, dynamicPalette)
			if int(paletteIndex) >= len(dynamicPalette) {
				return C.CString(fmt.Sprintf("palette index %d out of bounds for palette size %d", paletteIndex, len(dynamicPalette)))
			}
			mappedColor := dynamicPalette[paletteIndex]
			palettizedViewImg.Set(x, y, color.RGBA{mappedColor.R, mappedColor.G, mappedColor.B, uint8(a_val >> 8)})
		}
	}

	var jpegBuf bytes.Buffer
	err = jpeg.Encode(&jpegBuf, palettizedViewImg, &jpeg.Options{Quality: 75})
	if err != nil {
		return C.CString(fmt.Sprintf("failed to encode processed image to JPEG: %v", err))
	}

	jpegBytes := jpegBuf.Bytes()
	*outputJPEG = (*C.char)(C.CBytes(jpegBytes))
	*outputJPEGLen = C.int(len(jpegBytes))

	*outputPaletteJson = C.CString(string(paletteJSONBytes))
	*outputPaletteJsonLen = C.int(len(paletteJSONBytes))

	return nil
}


//export CompressImageToCustomFormat
func CompressImageToCustomFormat(
	inputImageBytes unsafe.Pointer, inputImageLen C.int,
	targetWidth C.uint, targetHeight C.uint,
	numPaletteColors C.int, blurSigma C.float, quantizationShiftAmount C.uint,
	outputCompressedData **C.char, outputCompressedDataLen *C.int,
	outputPaletteJson **C.char, outputPaletteJsonLen *C.int,
) *C.char {
	goInputBytes := C.GoBytes(inputImageBytes, inputImageLen)

	img, _, err := image.Decode(bytes.NewReader(goInputBytes))
	if err != nil {
		return C.CString(fmt.Sprintf("CompressImage: failed to decode: %v", err))
	}

	_, dynamicPalette, paletteIndexedBytes, err := processAndPaletteIndex(
		img, uint(targetWidth), uint(targetHeight),
		int(numPaletteColors), float32(blurSigma), uint(quantizationShiftAmount),
	)
	if err != nil {
		return C.CString(fmt.Sprintf("CompressImage: processing error: %v", err))
	}

	paletteJSONBytes, err := json.Marshal(dynamicPalette)
	if err != nil {
		return C.CString(fmt.Sprintf("CompressImage: failed to marshal palette: %v", err))
	}

	var zlibCompressedData bytes.Buffer
	zlibWriter, _ := zlib.NewWriterLevel(&zlibCompressedData, zlib.BestCompression)
	if _, err := zlibWriter.Write(paletteIndexedBytes); err != nil {
		zlibWriter.Close()
		return C.CString(fmt.Sprintf("CompressImage: zlib compression failed: %v", err))
	}
	if err := zlibWriter.Close(); err != nil {
		return C.CString(fmt.Sprintf("CompressImage: failed to close zlib writer: %v", err))
	}

	compressedBytes := zlibCompressedData.Bytes()
	*outputCompressedData = (*C.char)(C.CBytes(compressedBytes))
	*outputCompressedDataLen = C.int(len(compressedBytes))

	*outputPaletteJson = C.CString(string(paletteJSONBytes))
	*outputPaletteJsonLen = C.int(len(paletteJSONBytes))

	return nil
}

func decompressCore(
	compressedData []byte, paletteJsonStr string,
	targetWidth, targetHeight uint, numPaletteColorsIn int,
	grayscale bool,
) ([]byte, error) {
	var palette []RGB
	if err := json.Unmarshal([]byte(paletteJsonStr), &palette); err != nil {
		return nil, fmt.Errorf("failed to unmarshal palette JSON: %w", err)
	}

	if len(palette) == 0 && numPaletteColorsIn > 0 {
		return nil, fmt.Errorf("palette is empty but numPaletteColors = %d suggests it shouldn't be", numPaletteColorsIn)
	}
	numActualPaletteColors := len(palette)
	if numActualPaletteColors == 0 {
		numActualPaletteColors = numPaletteColorsIn
	}
	if numActualPaletteColors == 0 {
	    return nil, fmt.Errorf("cannot determine number of palette colors for bpp calculation")
	}


	bitsPerPixel, err := getBitsPerPixel(numActualPaletteColors)
	if err != nil {
		return nil, fmt.Errorf("decompress: error calculating bits per pixel: %w", err)
	}

	zlibReader, err := zlib.NewReader(bytes.NewReader(compressedData))
	if err != nil {
		return nil, fmt.Errorf("failed to create zlib reader: %w", err)
	}
	defer zlibReader.Close()

	paletteIndexedData, err := io.ReadAll(zlibReader)
	if err != nil {
		return nil, fmt.Errorf("failed to decompress zlib data: %w", err)
	}

	var decompressedImg image.Image
	var outputBuffer bytes.Buffer

	if grayscale {
		grayImg := image.NewGray(image.Rect(0, 0, int(targetWidth), int(targetHeight)))
		decompressedImg = grayImg
	} else {
		rgbaImg := image.NewRGBA(image.Rect(0, 0, int(targetWidth), int(targetHeight)))
		decompressedImg = rgbaImg
	}

	var bitBuffer uint32
	var bitsInBuffer uint8
	byteIdx := 0
	totalPixels := int(targetWidth * targetHeight)
	pixelsDecoded := 0
	pixelMask := uint32((1 << bitsPerPixel) - 1)

	for y := 0; y < int(targetHeight); y++ {
		for x := 0; x < int(targetWidth); x++ {
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
						return nil, fmt.Errorf("unexpected end of compressed data. Decoded %d of %d pixels", pixelsDecoded, totalPixels)
					}
					break
				}
			}

			if bitsInBuffer >= uint8(bitsPerPixel) {
				shiftAmount := bitsInBuffer - uint8(bitsPerPixel)
				paletteIndex := uint8((bitBuffer >> shiftAmount) & pixelMask)

				if int(paletteIndex) >= len(palette) {
					return nil, fmt.Errorf("decoded palette index %d out of bounds for palette size %d", paletteIndex, len(palette))
				}

				if grayscale {
					var grayscaleValue uint8
					if numActualPaletteColors <= 1 {
						grayscaleValue = 0
					} else {
						grayscaleValue = 255 - uint8(math.Round(float64(paletteIndex)*255.0/float64(numActualPaletteColors-1)))
					}
					decompressedImg.(draw.Image).Set(x, y, color.Gray{Y: grayscaleValue})
				} else {
					pColor := palette[paletteIndex]
					decompressedImg.(draw.Image).Set(x, y, color.RGBA{pColor.R, pColor.G, pColor.B, 255})
				}
				pixelsDecoded++

				bitsInBuffer -= uint8(bitsPerPixel)
				if bitsInBuffer > 0 {
					bitBuffer &= (1 << bitsInBuffer) - 1
				} else {
					bitBuffer = 0
				}
			} else if pixelsDecoded < totalPixels {
				log.Printf("Warning: Insufficient bits remaining for pixel %d. Bits needed: %d, bits have: %d. File might be truncated.", pixelsDecoded, bitsPerPixel, bitsInBuffer)
			}
		}
		if pixelsDecoded >= totalPixels {
			break
		}
	}

	if pixelsDecoded < totalPixels {
		log.Printf("Warning: Decoded only %d of %d pixels. Input file might be truncated or corrupted.", pixelsDecoded, totalPixels)
	}

	if err := png.Encode(&outputBuffer, decompressedImg); err != nil {
		return nil, fmt.Errorf("failed to encode decompressed image to PNG: %w", err)
	}
	return outputBuffer.Bytes(), nil
}

//export DecompressCustomFormatToOutputImage
func DecompressCustomFormatToOutputImage(
	compressedData unsafe.Pointer, compressedDataLen C.int,
	paletteJsonCStr *C.char,
	targetWidth C.uint, targetHeight C.uint, numPaletteColors C.int,
	outputImage **C.char, outputImageLen *C.int,
) *C.char {
	goCompressedData := C.GoBytes(compressedData, compressedDataLen)
	paletteJsonStr := C.GoString(paletteJsonCStr)

	pngBytes, err := decompressCore(goCompressedData, paletteJsonStr, uint(targetWidth), uint(targetHeight), int(numPaletteColors), false)
	if err != nil {
		return C.CString(fmt.Sprintf("DecompressImage: %v", err))
	}

	*outputImage = (*C.char)(C.CBytes(pngBytes))
	*outputImageLen = C.int(len(pngBytes))
	return nil
}

//export DecompressCustomFormatToOutputGrayscaleImage
func DecompressCustomFormatToOutputGrayscaleImage(
	compressedData unsafe.Pointer, compressedDataLen C.int,
	paletteJsonCStr *C.char,
	targetWidth C.uint, targetHeight C.uint, numPaletteColors C.int,
	outputImage **C.char, outputImageLen *C.int,
) *C.char {
	goCompressedData := C.GoBytes(compressedData, compressedDataLen)
	paletteJsonStr := C.GoString(paletteJsonCStr)

	pngBytes, err := decompressCore(goCompressedData, paletteJsonStr, uint(targetWidth), uint(targetHeight), int(numPaletteColors), true)
	if err != nil {
		return C.CString(fmt.Sprintf("DecompressGrayscaleImage: %v", err))
	}

	*outputImage = (*C.char)(C.CBytes(pngBytes))
	*outputImageLen = C.int(len(pngBytes))
	return nil
}

func main() {}