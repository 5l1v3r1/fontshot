package fontshot

import (
	"sort"

	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec/anyvec32"
)

// PretrainSamples is an anyff.SampleList for pre-training
// a model on classification.
type PretrainSamples struct {
	mapping map[rune]int
	samples []*Sample
}

// NewPretrainSamples creates a new PretrainSamples.
func NewPretrainSamples(s []*Sample) *PretrainSamples {
	r := []rune{}
	pres := map[rune]bool{}
	for _, x := range s {
		if !pres[x.Label] {
			pres[x.Label] = true
			r = append(r, x.Label)
		}
	}
	sort.Slice(r, func(i int, j int) bool {
		return r[i] < r[j]
	})
	mapping := map[rune]int{}
	for i, ru := range r {
		mapping[ru] = i
	}
	return &PretrainSamples{
		mapping: mapping,
		samples: s,
	}
}

// Len returns the number of samples.
func (p *PretrainSamples) Len() int {
	return len(p.samples)
}

// Swap swaps two samples.
func (p *PretrainSamples) Swap(i, j int) {
	s := p.samples
	s[i], s[j] = s[j], s[i]
}

// Slice gets a sub-set of the samples.
func (p *PretrainSamples) Slice(i, j int) anysgd.SampleList {
	return &PretrainSamples{
		samples: append([]*Sample{}, p.samples[i:j]...),
		mapping: p.mapping,
	}
}

// GetSample gets a feed-forward classification sample.
func (p *PretrainSamples) GetSample(idx int) (*anyff.Sample, error) {
	cm := p.ClassMap()
	outVec := make([]float32, len(cm))
	outVec[cm[p.samples[idx].Label]] = 1

	in, err := vectorForSample(anyvec32.CurrentCreator(), p.samples[idx])
	if err != nil {
		return nil, err
	}

	return &anyff.Sample{
		Input:  in,
		Output: anyvec32.MakeVectorData(outVec),
	}, nil
}

// ClassMap returns a mapping from labels to their
// corresponding class indices.
func (p *PretrainSamples) ClassMap() map[rune]int {
	return p.mapping
}
