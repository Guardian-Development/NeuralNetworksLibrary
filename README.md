# Neural Networks Library
### Neural Network implementation in .NET Core

## Description
A Library that implements Neural Networks. Example initialisation of a Neural Network would be: 

```csharp
var neuralNetwork = NeuralNetwork.For()
            .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
            .WithHiddenLayer(neuronCount: 3, activationType: ActivationType.TanH)
            .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
            .Build();

TrainingController<BackPropagation>
	.For(BackPropagation.WithConfiguration(neuralNetwork, learningRate: 0.4, momentum: 0.9))
	.TrainForEpochsOrErrorThresholdMet(GetXorTrainingData(), maximumEpochs: 3000, errorThreshold: 0.1);
```
## Next Steps
- Add back propagation (this should drive the final model for neurons and layers etc)
- Once implemented look at adding testing with the XOR dataset 
- Once tested look at code cleanup/what needs to be exposed publically (builder for neural network)
- Add unit testing around library (would usually come earlier but need to see it work first to get goal in head)
- Add performance tuning 
- Add documentation 
- Deploy to Nuget 