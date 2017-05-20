# Neural Networks Library
## Neural Network implementation in .NET Core

# Description
A Library that implements Neural Networks. Example initialisation of a Neural Network would be: 

```csharp
 var neuralNetwork = NeuralNetwork.Create()
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 3, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid);
```
