package fontshot

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	var m Model
	serializer.RegisterTypedDeserializer(m.SerializerType(), DeserializeModel)
}

// A Model encapsulates the learner and the classifier.
type Model struct {
	// Learner takes an example as input and produces a
	// knowledge vector as output.
	Learner anynet.Layer

	// Mix knowledge+input pairs for Classifier.
	Mixer anynet.Mixer

	// Turn result from Mixer into pre-sigmoid probability.
	Classifier anynet.Layer
}

// NewModel creates a new, untrained Model.
func NewModel(c anyvec.Creator, knowledgeSize int) *Model {
	convCode := `
	Input(w=54, h=54, d=1)
	Conv(w=3, h=3, n=8, sx=2, sy=2)
	BatchNorm
	ReLU
	Conv(w=3, h=3, n=16, sx=2, sy=2)
	BatchNorm
	ReLU
	Conv(w=3, h=3, n=32, sx=2, sy=2)
	BatchNorm
	ReLU
	FC(out=128)
	Tanh
	`
	learnerLayer, err := anyconv.FromMarkup(c, convCode)
	if err != nil {
		panic(err)
	}
	mixerLayer, err := anyconv.FromMarkup(c, convCode)
	if err != nil {
		panic(err)
	}
	return &Model{
		Learner: append(learnerLayer.(anynet.Net),
			anynet.NewFC(c, 128, knowledgeSize)),
		Mixer: &anynet.AddMixer{
			In1: anynet.Net{
				anynet.NewFC(c, knowledgeSize, 128),
				anynet.Tanh,
			},
			In2: mixerLayer,
			Out: anynet.Net{
				anynet.Tanh,
				anynet.NewFC(c, 128, 128),
			},
		},
		Classifier: anynet.NewFC(c, 128, 1),
	}
}

// DeserializeModel deserializes a Model.
func DeserializeModel(d []byte) (*Model, error) {
	var m Model
	err := serializer.DeserializeAny(d, &m.Learner, &m.Mixer, &m.Classifier)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Model", err)
	}
	return &m, nil
}

// Apply looks at the examples and then classifies a batch
// of new images based on the examples.
// The resulting classifications are pre-sigmoid
// probabilities.
func (m *Model) Apply(examples, inputs anydiff.Res, n int) anydiff.Res {
	knowledge := m.Learner.Apply(examples, n)
	mixed := m.Mixer.Mix(knowledge, inputs, n)
	return m.Classifier.Apply(mixed, n)
}

// Parameters returns all the parameters of the model.
func (m *Model) Parameters() []*anydiff.Var {
	var res []*anydiff.Var
	for _, x := range []interface{}{m.Learner, m.Mixer, m.Classifier} {
		if p, ok := x.(anynet.Parameterizer); ok {
			res = append(res, p.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a Model with the serializer package.
func (m *Model) SerializerType() string {
	return "github.com/unixpickle/fontshot.Model"
}

// Serialize serializes the model.
func (m *Model) Serialize() (d []byte, err error) {
	defer essentials.AddCtxTo("serialize Model", &err)
	return serializer.SerializeAny(m.Learner, m.Mixer, m.Classifier)
}
