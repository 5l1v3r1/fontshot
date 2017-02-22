package fontshot

import (
	"fmt"
	"image"
	_ "image/png"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// ImageSize is the required size of font image samples.
const ImageSize = 54

// A Sample is a single image paired with its label.
type Sample struct {
	ImagePath string
	Label     rune
}

// ReadSamples processes a font-dump output directory and
// produces a Samples instance.
//
// The directory should have a directory for each font.
// Inside a font directory, the directories "lowercase",
// "uppercase", and "digits" contain digit images.
// Directories may be missing for certain fonts.
func ReadSamples(path string) (samples []*Sample, err error) {
	defer essentials.AddCtxTo("read samples", &err)
	listing, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}
	for _, item := range listing {
		if !item.IsDir() {
			continue
		}
		subPath := filepath.Join(path, item.Name())
		subListing, err := ioutil.ReadDir(subPath)
		if err != nil {
			return nil, err
		}
		for _, subItem := range subListing {
			supported := map[string]bool{"lowercase": true, "uppercase": true,
				"digits": true}
			if !subItem.IsDir() || !supported[subItem.Name()] {
				continue
			}
			err := listImages(filepath.Join(subPath, subItem.Name()), &samples)
			if err != nil {
				return nil, err
			}
		}
	}
	return
}

func listImages(path string, dest *[]*Sample) error {
	listing, err := ioutil.ReadDir(path)
	if err != nil {
		return err
	}
	for _, item := range listing {
		if filepath.Ext(item.Name()) == ".png" && len(item.Name()) == 5 {
			*dest = append(*dest, &Sample{
				ImagePath: filepath.Join(path, item.Name()),
				Label:     rune(item.Name()[0]),
			})
		}
	}
	return nil
}

func vectorForSample(c anyvec.Creator, s *Sample) (anyvec.Vector, error) {
	r, err := os.Open(s.ImagePath)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, err
	}

	if img.Bounds().Dx() != ImageSize || img.Bounds().Dy() != ImageSize {
		return nil, fmt.Errorf("expected %dx%d but got %dx%d", ImageSize, ImageSize,
			img.Bounds().Dx(), img.Bounds().Dy())
	}

	buf := make([]float64, 0, ImageSize*ImageSize)
	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			_, _, _, a := img.At(x, y).RGBA()
			buf = append(buf, float64(a)/0xffff)
		}
	}

	return c.MakeVectorData(c.MakeNumericList(buf)), nil
}
