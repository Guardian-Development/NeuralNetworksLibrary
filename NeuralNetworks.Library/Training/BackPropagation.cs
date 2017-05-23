using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Layers;

namespace NeuralNetworks.Library.Training
{
    public sealed class BackPropagation : ITrainNeuralNetworks
    {
        private readonly NeuralNetwork neuralNetwork;
        private readonly double learningRate;
        private readonly double momentum;

        private BackPropagation(NeuralNetwork neuralNetwork, double learningRate, double momentum)
        {
            this.neuralNetwork = neuralNetwork;
            this.learningRate = learningRate;
            this.momentum = momentum;
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
                Console.WriteLine($"EPOCH: {currentEpoch}. ERROR RATE: {currentErrorRate}");
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

                SetOutputLayerNeuronErrorRates(neuralNetwork.OutputLayer, predictedOutput, expectedOutput);

                neuralNetwork
                    .HiddenLayers
                    .Reverse()
                    .ToList()
                    .ForEach(SetHiddenLayerNeuronErrorRates);

                neuralNetwork
                    .OutputLayer
                    .Concat(neuralNetwork.HiddenLayers)
                    .Reverse()
                    .ToList()
                    .ForEach(layer => UpdateSynapseWeights(layer, synapseKnownDeltas));


                predictedOutput = neuralNetwork.MakePrediction(input);
                currentErrorRate += NeuralNetworkErrorRate(predictedOutput, expectedOutput);
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
                var errorRate = currentNeuron.ActivationFunction.Derivative(currentNeuron.SumOfInputValues) *
                                (predictedOutput[i] - expectedOutput[i]);
                currentNeuron.ErrorRate = errorRate;
            }
        }

        private void SetHiddenLayerNeuronErrorRates(HiddenLayer hiddenLayer)
        {
            foreach (var currentNeuron in hiddenLayer.Neurons)
            {
                var sumOfErrorsFedIntoNeuron =
                    GetSumOfErrorsNeuronContributesTo(hiddenLayer.NextLayer, currentNeuron);

                var neuronErrorRate = currentNeuron.ActivationFunction.Derivative(currentNeuron.SumOfInputValues);
                currentNeuron.ErrorRate = neuronErrorRate * sumOfErrorsFedIntoNeuron;
            }
        }

        private void UpdateSynapseWeights(Layer layer, IDictionary<Synapse, double?> synapseKnownDeltas)
        {
            foreach (var neuron in layer.Neurons)
            {
                foreach (var synapse in neuron.InputConnections)
                {
                    var delta = learningRate * neuron.ErrorRate * synapse.Source.Output;
                    synapseKnownDeltas.TryGetValue(synapse, out var knownDelta);

                    if (knownDelta.HasValue)
                    {
                        delta += momentum * knownDelta.Value; 
                    }

                    synapse.Weight = synapse.Weight - delta; 
                    synapseKnownDeltas[synapse] = delta;
                }
            }

        }

        private double GetSumOfErrorsNeuronContributesTo(Layer nextLayer, Neuron sourceNeuron) =>
            nextLayer.Neurons
                .Select(effectedNeuron => new
                {
                    effectedNeuron,
                    synapsesForSourceNeuron =
                    effectedNeuron.InputConnections.Where(synapse => synapse.Source == sourceNeuron)
                })
                .SelectMany(neuronWithSynapses =>
                    neuronWithSynapses
                        .synapsesForSourceNeuron
                        .Select(synapse => synapse.Weight * neuronWithSynapses.effectedNeuron.ErrorRate)
                ).Sum();

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