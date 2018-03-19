# Neural Networks Library
[![Build Status](https://travis-ci.org/Guardian-Development/NeuralNetworksLibrary.svg?branch=master)](https://travis-ci.org/Guardian-Development/NeuralNetworksLibrary)

### Neural Network implementation in .NET Core
https://www.nuget.org/packages/GuardianDevelopment.NeuralNetworks.Library/

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

## How To Use 
The below steps show how to use the Neural Network Library for solving a simple problem, the XOR problem. 

### The Problem Overview 
We will be looking to create a Neural Network that when given values corresponding to the inputs to the XOR operation they produce the correct output. The XOR problem can be defined for our use case as "when given 2 inputs of 0 or 1, if the inputs are not equal then return 1, else return 0." 

| Inputs/Outputs | 0 | 1 |
|:--------------:|:-:|:-:|
|        0       | 0 | 1 |
|        1       | 1 | 0 |

### Step 1: Choosing the shape of your Neural Network
When deciding on your Neural Network there are a few factors to consider. 
1. The shape of your input data. (In this example the shape is 2, as there are 2 data points we are inputting)
2. How many hidden layers you require. (In this example we require only 1 hidden layer, as we have a very simple relationship between input data and the output required, this type of network is also known as a Perceptron)
3. The shape of your output data. (In this example the shape is 1, as there is 1 data point we are predicting which can be 1 or 0)
4. The Activation Function you wish to use with each layer of your Neural Network. (This is a complex topic, but I will simplify it by saying it is often good practice to have hidden layers use the TanH Function and input and output layers use the Sigmoid Function)

With this information at hand you can now initialise a Neural Network as follows. 

```csharp
var neuralNetwork = NeuralNetwork.For(NeuralNetworkContext.MaximumPrecision)
                .WithInputLayer(neuronCount: 2, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 5, activationType: ActivationType.TanH)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();
```
The Neural Network Context allows the setting of the precision of the storage of the results of specific calculations within the Network. Maximum Precision, as the name implies, allows the greatest degree of precision. 

### Step 2: Training your Neural Network 
Once you have a Neural Network initialised you are in a good position to train your network. Network training requires you to have access to data that shows, with a set of inputs what the desired output looks like. This concept is encapsulated in the TrainingDataSet class. This allows you to tell a class that trains Neural Networks how to adjust the network to produce the desired outputs. 

#### Creating Training Data 
For our simple XOR problem the training data set is very simple and can be defined as shown: 

```csharp
private static List<TrainingDataSet> XorTrainingData()
{
    var inputs = new[]
    {
        new[] {0.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}
    };

    var outputs = new[]
    {
        new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}
    };

    return TrainingDataSetExtensions.BuildTrainingDataForAllInputs(inputs, outputs).ToList(); 
}
```

You can see that this is a direct representation of the table I displayed above. In more complex examples you won't have access to the entire range of data inputs and their respective outputs. 

It is important to note that the inputs and outputs should be the same shape as the Networks input layer and output layer. 

#### Creating the Training Controller
The Neural Network Library currently supports only the Back Propagation algorithm for training a Neural Network. However, it is open for extension and therefore you must wrap your training method in a controller which can then handle training the network. This can be done like so: 

```csharp
var controller = TrainingController.For(BackPropagation.WithConfiguration(neuralNetwork)
```

If you wish to use a Learning Rate or Momentum when training the Network these parameters can be passed into the configuration of the Back Propagation algorithm at this point.

- Learning rate is the speed in which your Network adopts new concepts, which by default is at a speed of 1. 
- Momentum is how fast your network attempts to reduce its error rate, this can allow for faster training but can also allow your network to overshoot the optimum solution. The default is a momentum of 0. 

#### Training the Network
Once you have a Training Controller you are in the position of allowing your Network to be trained. The Training Controller allows multiple ways of training the Network based on your goals. They can be seen below: 

```csharp
await controller.TrainForEpochsOrErrorThresholdMet(GetXorTrainingData(), maximumEpochs: 3000, errorThreshold: 0.1);
await controller.TrainForEpochs(GetXorTrainingData(), maximumEpochs: 5000); 

//WARNING: if your data does not correlate to the desired output to allow the error threshold to be met this will iterate infinitly
await controller.TrainForErrorThreshold(minimumErrorThreshold: 0.1); 
```

An Epoch is a single iteration through your Training Data set. 

### Making predictions 
Once you have a trained Network you are in a position to make accurate predictions on your data. To do this: 

```csharp
var prediction = neuralNetwork.PredictionFor(new [] { 0.0, 1.0 }, ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions); 
```
You specify the Parallel Options the network should use when Multi Threading.
ParallelOptionsExtensions are an extension around this allowing you to create ParallelOptions configurations easily, if you wish. 

### Result
You have now created, trained and made predictions from a Neural Network. If you wish to see further examples of using the Network you can find some example solutions to known data sets here: 
[Known Data Set Solutions](./NeuralNetworks.Tests.IntegrationTests/DatasetCaseStudies/)
