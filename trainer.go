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
	Examples *anydiff.Const
	Inputs   *anydiff.Const
	Outputs  *anydiff.Const
}

// Trainer produces batches and computes gradients.
type Trainer struct {
	Model       *Model
	NumExamples int
	Samples     []*Sample

	// LastCost is the cost from the previous call to
	// Gradient.
	LastCost anyvec.Numeric
}

// Fetch produces a random batch.
// The length of s is used to determine the number of
// images to feed to the classifier.
func (t *Trainer) Fetch(s anysgd.SampleList) (batch anysgd.Batch, err error) {
	defer essentials.AddCtxTo("fetch samples", &err)

	class := t.randomClass()
	ex := t.randomExamples(class)
	inputs, outputs := t.randomInputs(class, s.Len())

	c := t.Model.Parameters()[0].Vector.Creator()
	exBatch, err := packedSamples(c, ex)
	if err != nil {
		return nil, err
	}
	inBatch, err := packedSamples(c, inputs)
	if err != nil {
		return nil, err
	}
	return &Batch{
		Examples: exBatch,
		Inputs:   inBatch,
		Outputs:  anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(outputs))),
	}, nil
}

// TotalCost computes the average cost over the batch.
func (t *Trainer) TotalCost(b *Batch) anydiff.Res {
	n := b.Outputs.Output().Len()
	outs := t.Model.Apply(b.Examples, b.Inputs, t.NumExamples, n)
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
	present := map[rune]int{}
	classes := []rune{}
	for _, x := range t.Samples {
		present[x.Label]++
		if present[x.Label] == t.NumExamples {
			classes = append(classes, x.Label)
		}
	}
	return classes[rand.Intn(len(classes))]
}

func (t *Trainer) randomExamples(class rune) []*Sample {
	options := t.samplesInClass(class)
	res := []*Sample{}
	for _, i := range rand.Perm(len(options))[:t.NumExamples] {
		res = append(res, options[i])
	}
	return res
}

func (t *Trainer) samplesInClass(class rune) []*Sample {
	return t.samplesForCond(func(s *Sample) bool {
		return s.Label == class
	})
}

func (t *Trainer) randomInputs(class rune, num int) (inputs []*Sample, outputs []float64) {
	inClass := t.samplesInClass(class)
	outClass := t.samplesForCond(func(s *Sample) bool {
		return s.Label != class
	})
	for i := 0; i < num; i++ {
		var s *Sample
		if rand.Intn(2) == 0 {
			s = outClass[rand.Intn(len(outClass))]
			outputs = append(outputs, 0)
		} else {
			outputs = append(outputs, 1)
			s = inClass[rand.Intn(len(inClass))]
		}
		inputs = append(inputs, s)
	}
	return
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
