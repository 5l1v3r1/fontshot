package main

import (
	"flag"
	"log"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/fontshot"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	var modelPath string
	var sampleDir string
	var validation string
	var stepSize float64
	var batchSize int
	var knowledgeSize int
	var pretrain bool

	flag.StringVar(&modelPath, "model", "model_out", "model file")
	flag.StringVar(&sampleDir, "samples", "", "sample directory")
	flag.StringVar(&validation, "validation", "aBcD94", "validation runes")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
	flag.IntVar(&batchSize, "batch", 64, "number of sets per batch")
	flag.IntVar(&knowledgeSize, "knowledge", 16, "size of learned knowledge vectors")
	flag.BoolVar(&pretrain, "pretrain", false, "train on classification")

	flag.Parse()

	if sampleDir == "" {
		essentials.Die("Missing -samples flag. See -help for more.")
	}

	samples, err := fontshot.ReadSamples(sampleDir)
	if err != nil {
		essentials.Die(err)
	}
	validSet, trainSet := fontshot.Partition(samples, validation)
	log.Printf("Samples: %d training and %d validation", len(trainSet), len(validSet))

	var model *fontshot.Model
	if err := serializer.LoadAny(modelPath, &model); err != nil {
		log.Println("Creating new model...")
		model = fontshot.NewModel(anyvec32.CurrentCreator(), knowledgeSize)
	} else {
		log.Println("Loaded model.")
	}

	if pretrain {
		trainClassifier(model, trainSet, stepSize, batchSize)
	} else {
		train(model, validSet, trainSet, stepSize, batchSize)
	}

	if err := serializer.SaveAny(modelPath, model); err != nil {
		essentials.Die(err)
	}
}

func train(model *fontshot.Model, validSet, trainSet []*fontshot.Sample,
	stepSize float64, batchSize int) {
	tr := &fontshot.Trainer{
		Model:   model,
		Samples: trainSet,
	}

	var iter int
	sgd := &anysgd.SGD{
		Fetcher:    tr,
		Gradienter: tr,
		Samples:    anysgd.LengthSampleList(batchSize),
		BatchSize:  batchSize,
		Rater:      anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			tr1 := *tr
			tr1.Samples = validSet
			vBatch, err := tr1.Fetch(anysgd.LengthSampleList(batchSize))
			if err != nil {
				essentials.Die(err)
			}
			vCost := anyvec.Sum(tr.TotalCost(vBatch.(*fontshot.Batch)).Output())
			log.Printf("iter %d: cost=%v validation=%v", iter, tr.LastCost, vCost)
			iter++
		},
	}

	if err := sgd.Run(rip.NewRIP().Chan()); err != nil {
		essentials.Die(err)
	}
}

func trainClassifier(model *fontshot.Model, trainSet []*fontshot.Sample,
	stepSize float64, batchSize int) {
	samples := fontshot.NewPretrainSamples(trainSet)
	learner := model.Learner.(anynet.Net)
	classifier := append(append(anynet.Net{}, learner[:len(learner)-1]...),
		anynet.NewFC(anyvec32.CurrentCreator(), 128, len(samples.ClassMap())),
		anynet.LogSoftmax)
	log.Println("Training with", len(samples.ClassMap()), "classes...")
	tr := &anyff.Trainer{
		Net:     classifier,
		Cost:    anynet.DotCost{},
		Params:  classifier.Parameters(),
		Average: true,
	}
	var iter int
	sgd := &anysgd.SGD{
		Fetcher:    tr,
		Gradienter: tr,
		Samples:    samples,
		BatchSize:  batchSize,
		Rater:      anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			log.Printf("iter %d: cost=%v", iter, tr.LastCost)
			iter++
		},
	}
	if err := sgd.Run(rip.NewRIP().Chan()); err != nil {
		essentials.Die(err)
	}

	// Use the pre-trained knowledge for the classifier as
	// well as for the learner.
	copy(model.Mixer.(*anynet.AddMixer).In2.(anynet.Net), classifier[:len(classifier)-2])
}
