# Neural Networks Library
[![Build Status](https://travis-ci.org/Guardian-Development/NeuralNetworksLibrary.svg?branch=master)](https://travis-ci.org/Guardian-Development/NeuralNetworksLibrary)

### Neural Network implementation in .NET Core
https://preview.nuget.org/packages/GuardianDevelopment.NeuralNetworks.Library/

## Description
A Library that implements Neural Networks. Example initialisation of a Neural Network would be: 

```csharp
var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

await TrainingController
	.For(BackPropagation.WithConfiguration(neuralNetwork, learningRate: 0.4, momentum: 0.9))
	.TrainForEpochsOrErrorThresholdMet(GetXorTrainingData(), maximumEpochs: 3000, errorThreshold: 0.1);
```
## Next Steps
- Add documentation