package fontshot

import (
	"io/ioutil"
	"path/filepath"

	"github.com/unixpickle/essentials"
)

// Samples maps runes to image sample paths.
type Samples map[rune][]string

// ReadSamples processes a font-dump output directory and
// produces a Samples instance.
//
// The directory should have a directory for each font.
// Inside a font directory, the directories "lowercase",
// "uppercase", and "digits" contain digit images.
// Directories may be missing for certain fonts.
func ReadSamples(path string) (samples Samples, err error) {
	defer essentials.AddCtxTo("read samples", &err)
	samples = Samples{}
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
			err := listImages(filepath.Join(subPath, subItem.Name()), samples)
			if err != nil {
				return nil, err
			}
		}
	}
	return
}

func listImages(path string, dest Samples) error {
	listing, err := ioutil.ReadDir(path)
	if err != nil {
		return err
	}
	for _, item := range listing {
		if filepath.Ext(item.Name()) == ".png" && len(item.Name()) == 5 {
			r := rune(item.Name()[0])
			p := filepath.Join(path, item.Name())
			dest[r] = append(dest[r], p)
		}
	}
	return nil
}
