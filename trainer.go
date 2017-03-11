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

	var examples []angledSample
	var inputs []angledSample
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

func (t *Trainer) randomClass() angledClass {
	present := map[rune]bool{}
	classes := []rune{}
	for _, x := range t.Samples {
		if !present[x.Label] {
			present[x.Label] = true
			classes = append(classes, x.Label)
		}
	}
	return angledClass{
		Char:  classes[rand.Intn(len(classes))],
		Angle: rand.Intn(4),
	}
}

func (t *Trainer) randomExample(class angledClass) angledSample {
	options := t.samplesInClass(class)
	return options[rand.Intn(len(options))]
}

func (t *Trainer) randomInput(class angledClass) (angledSample, float64) {
	if rand.Intn(2) == 1 {
		return t.randomExample(class), 1
	} else {
		s := t.samplesNotInClass(class)
		return s[rand.Intn(len(s))], 0
	}
}

func (t *Trainer) samplesInClass(class angledClass) []angledSample {
	return t.samplesForCond(func(a angledSample) bool {
		return a.Sample.Label == class.Char && a.Angle == a.Angle
	})
}

func (t *Trainer) samplesNotInClass(class angledClass) []angledSample {
	return t.samplesForCond(func(a angledSample) bool {
		return a.Sample.Label != class.Char || a.Angle != a.Angle
	})
}

func (t *Trainer) samplesForCond(f func(s angledSample) bool) []angledSample {
	res := []angledSample{}
	for _, x := range t.Samples {
		for _, angle := range []int{0, 1, 2, 3} {
			s := angledSample{Sample: x, Angle: angle}
			if f(s) {
				res = append(res, s)
			}
		}
	}
	return res
}

func packedSamples(c anyvec.Creator, samples []angledSample) (*anydiff.Const, error) {
	var parts []anyvec.Vector
	for _, x := range samples {
		vec, err := vectorForSample(c, x.Sample, x.Angle)
		if err != nil {
			return nil, err
		}
		parts = append(parts, vec)
	}
	return anydiff.NewConst(c.Concat(parts...)), nil
}

type angledSample struct {
	Sample *Sample
	Angle  int
}

type angledClass struct {
	Char  rune
	Angle int
}
