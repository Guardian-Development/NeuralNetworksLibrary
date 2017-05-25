using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Layers;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training
{
    public sealed class BackPropagation : ITrainNeuralNetworks
    {
        private static ILogger Log => LoggerProvider.For<BackPropagation>();

        private readonly NeuralNetwork neuralNetwork;
        private readonly double learningRate;
        private readonly double momentum;

        private BackPropagation(NeuralNetwork neuralNetwork, double learningRate, double momentum)
        {
            this.neuralNetwork = neuralNetwork;
            this.learningRate = learningRate;
            this.momentum = momentum;
            Log.LogDebug($"{nameof(BackPropagation)} created with. LearningRate: {learningRate}. Momentum: {momentum}");
        }

        public void TrainNetwork(
            double[][] trainingInputs,
            double[][] expectedOutputs,
            int epochs = 100, 
            double errorThreshold = 0.0001)
        {
            ValidateTrainingInputsWithExepctedOutputs(trainingInputs, expectedOutputs);

            var currentEpoch = 1;
            double currentErrorRate = 100;
            while (currentEpoch <= epochs && currentErrorRate >= errorThreshold)
            {
                currentErrorRate = ExecuteSingleEpoch(trainingInputs, expectedOutputs);
                Log.LogInformation($"{nameof(currentEpoch)}: {currentEpoch}. {nameof(currentErrorRate)}: {currentErrorRate}");
                currentEpoch++; 
            }
        }

        private double ExecuteSingleEpoch(double[][] trainingInputs,double[][] expectedOutputs)
        {
            var synapseKnownDeltas = new Dictionary<Synapse, double?>();
            double currentErrorRate = 0;

            for (var i = 0; i < trainingInputs.Length; i++)
            {
                MatchTrainingInputToExpectedOutput(i,
                    trainingInputs,
                    expectedOutputs,
                    out var input,
                    out var expectedOutput);

                var predictedOutput = neuralNetwork.MakePrediction(input);
                Log.LogDebug($"{nameof(ExecuteSingleEpoch)}{nameof(predictedOutput)}: {predictedOutput.LogArray()}");

                SetOutputLayerNeuronErrorRates(neuralNetwork.OutputLayer, predictedOutput, expectedOutput);

                neuralNetwork
                    .HiddenLayers
                    .Reverse()
                    .ToList()
                    .ForEach(SetHiddenLayerNeuronErrorRates);

                neuralNetwork
                    .OutputLayer
                    .Concat(neuralNetwork.HiddenLayers.Reverse())
                    .ToList()
                    .ForEach(layer => UpdateSynapseWeights(layer, synapseKnownDeltas));


                predictedOutput = neuralNetwork.MakePrediction(input);
                Log.LogDebug($"{nameof(ExecuteSingleEpoch)}{nameof(predictedOutput)}: {predictedOutput.LogArray()}");

                currentErrorRate += NeuralNetworkErrorRate(predictedOutput, expectedOutput);
                Log.LogDebug($"{nameof(ExecuteSingleEpoch)}{nameof(currentErrorRate)}: {currentErrorRate}");
            }

            return currentErrorRate; 
        }

        private void SetOutputLayerNeuronErrorRates(
            OutputLayer outputLayer,
            double[] predictedOutput,
            double[] expectedOutput)
        {
            for (var i = 0; i < outputLayer.Neurons.Length; i++)
            {
                var currentNeuron = outputLayer.Neurons[i];
                var errorRate = currentNeuron.ActivationFunction.Derivative(currentNeuron.Output) *
                                (predictedOutput[i] - expectedOutput[i]);

                Log.LogDebug($"{nameof(SetOutputLayerNeuronErrorRates)} {nameof(errorRate)}: {errorRate}");

                currentNeuron.ErrorRate = errorRate;

                Log.LogDebug($"{nameof(SetOutputLayerNeuronErrorRates)} Current Neuron Error Rate Set: {currentNeuron.ErrorRate}");
            }
        }

        private void SetHiddenLayerNeuronErrorRates(HiddenLayer hiddenLayer)
        {
            foreach (var currentNeuron in hiddenLayer.Neurons)
            {
                var neuronErrorRate =
                    CalculateNeuronErrorRate(hiddenLayer.NextLayer, currentNeuron);
                Log.LogDebug($"{nameof(neuronErrorRate)}: {neuronErrorRate}");

                currentNeuron.ErrorRate = neuronErrorRate;
                Log.LogDebug($"Current Neuron Error Rate: {currentNeuron.ErrorRate}");
            }
        }

        private void UpdateSynapseWeights(Layer layer, IDictionary<Synapse, double?> synapseKnownDeltas)
        {
            foreach (var neuron in layer.Neurons)
            {
                foreach (var synapse in neuron.InputConnections)
                {
                    var delta = learningRate * neuron.ErrorRate * synapse.Source.Output;
                    Log.LogDebug($"{nameof(synapse)} {nameof(delta)}: {delta}");

                    synapseKnownDeltas.TryGetValue(synapse, out var knownDelta);
                    Log.LogDebug($"{nameof(knownDelta)} : {knownDelta}");

                    if (knownDelta.HasValue)
                    {
                        delta += momentum * knownDelta.Value;
                        Log.LogDebug($"{nameof(delta)} : {delta}");
                    }

                    Log.LogDebug($"Pre Synapse Weight : {synapse.Weight}");

                    synapse.Weight = synapse.Weight - delta; 
                    synapseKnownDeltas[synapse] = delta;

                    Log.LogDebug($"Post Synapse Weight : {synapse.Weight}");
                }
            }

        }

        private double CalculateNeuronErrorRate(Layer nextLayer, Neuron sourceNeuron)
        {
            var neuronError = sourceNeuron.ActivationFunction.Derivative(sourceNeuron.Output);

            var sumOfErrors = nextLayer.Neurons
                .Select(effectedNeuron => new
                {
                    effectedNeuron,
                    effectedNeuronSynapse =
                    effectedNeuron.InputConnections.First(synapse => synapse.Source == sourceNeuron)
                })
                .Select(effectedNeuronWithSynapse =>
                    effectedNeuronWithSynapse.effectedNeuronSynapse.Weight * effectedNeuronWithSynapse.effectedNeuron.ErrorRate)
                .Sum();
            var result = sumOfErrors * neuronError; 
            return result; 
        }

        private void ValidateTrainingInputsWithExepctedOutputs(
            double[][] trainingInputs,
            double[][] expectedOutputs)
        {
            if (trainingInputs.Length != expectedOutputs.Length)
            {
                throw new ArgumentException(
                    $"{nameof(trainingInputs)} must be the same length as {expectedOutputs}");
            }
        }

        private void MatchTrainingInputToExpectedOutput(
            int index,
            double[][] trainingInputs,
            double[][] correspondingOutputs,
            out double[] inputForIndex,
            out double[] outputForIndex)
        {
            inputForIndex = trainingInputs[index];
            outputForIndex = correspondingOutputs[index];
        }

        public double NeuralNetworkErrorRate(double[] actualResult, double[] expectedResult) =>
            expectedResult
                .Select((expected, i) => Math.Pow(expected - actualResult[i], 2))
                .Sum() / 2;

        public static BackPropagation For(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}