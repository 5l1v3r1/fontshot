package fontshot

import (
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

// Batch is a batch of examples, inputs, and classifier
// labels.
type Batch struct {
	N        int
	Examples *anydiff.Const
	Inputs   *anydiff.Const
	Outputs  *anydiff.Const
}

// Trainer produces batches and computes gradients.
type Trainer struct {
	Model   *Model
	Samples []*Sample

	// LastCost is the cost from the previous call to
	// Gradient.
	LastCost anyvec.Numeric
}

// Fetch produces a random batch.
// The length of s is used to determine the batch size.
func (t *Trainer) Fetch(s anysgd.SampleList) (batch anysgd.Batch, err error) {
	defer essentials.AddCtxTo("fetch samples", &err)

	var examples []*Sample
	var inputs []*Sample
	var outputs []float64

	for i := 0; i < s.Len(); i++ {
		class := t.randomClass()
		examples = append(examples, t.randomExample(class))
		i, o := t.randomInput(class)
		inputs = append(inputs, i)
		outputs = append(outputs, o)
	}

	c := t.Model.Parameters()[0].Vector.Creator()
	exBatch, err := packedSamples(c, examples)
	if err != nil {
		return nil, err
	}
	inBatch, err := packedSamples(c, inputs)
	if err != nil {
		return nil, err
	}
	return &Batch{
		N:        s.Len(),
		Examples: exBatch,
		Inputs:   inBatch,
		Outputs:  anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(outputs))),
	}, nil
}

// TotalCost computes the average cost over the batch.
func (t *Trainer) TotalCost(b *Batch) anydiff.Res {
	outs := t.Model.Apply(b.Examples, b.Inputs, b.N)
	costFunc := anynet.SigmoidCE{Average: true}
	cost := costFunc.Cost(b.Outputs, outs, 1)
	return cost
}

// Gradient computes the gradient for the batch.
func (t *Trainer) Gradient(b anysgd.Batch) anydiff.Grad {
	res := anydiff.NewGrad(t.Model.Parameters()...)

	cost := t.TotalCost(b.(*Batch))
	t.LastCost = anyvec.Sum(cost.Output())

	c := cost.Output().Creator()
	upstream := c.MakeVectorData(c.MakeNumericList([]float64{1}))
	cost.Propagate(upstream, res)

	return res
}

func (t *Trainer) randomClass() rune {
	present := map[rune]bool{}
	classes := []rune{}
	for _, x := range t.Samples {
		if !present[x.Label] {
			present[x.Label] = true
			classes = append(classes, x.Label)
		}
	}
	return classes[rand.Intn(len(classes))]
}

func (t *Trainer) randomExample(class rune) *Sample {
	options := t.samplesInClass(class)
	return options[rand.Intn(len(options))]
}

func (t *Trainer) randomInput(class rune) (*Sample, float64) {
	if rand.Intn(2) == 1 {
		return t.randomExample(class), 1
	} else {
		s := t.samplesForCond(func(x *Sample) bool {
			return x.Label != class
		})
		return s[rand.Intn(len(s))], 0
	}
}

func (t *Trainer) samplesInClass(class rune) []*Sample {
	return t.samplesForCond(func(s *Sample) bool {
		return s.Label == class
	})
}

func (t *Trainer) samplesForCond(f func(s *Sample) bool) []*Sample {
	res := []*Sample{}
	for _, x := range t.Samples {
		if f(x) {
			res = append(res, x)
		}
	}
	return res
}

func packedSamples(c anyvec.Creator, samples []*Sample) (*anydiff.Const, error) {
	var parts []anyvec.Vector
	for _, x := range samples {
		vec, err := vectorForSample(c, x)
		if err != nil {
			return nil, err
		}
		parts = append(parts, vec)
	}
	return anydiff.NewConst(c.Concat(parts...)), nil
}
