using System;
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
            double errorThreshold = 0.0001)
        {
            ValidateTrainingInputsWithExepctedOutputs(trainingInputs, expectedOutputs);

            for (var i = 0; i < trainingInputs.Length; i++)
            {
                MatchTrainingInputToExpectedOutput(i, 
                    trainingInputs, 
                    expectedOutputs, 
                    out var input, 
                    out var expectedOutput);
;
                var predictedOutput = neuralNetwork.MakePrediction(input);

                SetOutputLayerNeuronErrorRates(neuralNetwork.OutputLayer, predictedOutput, expectedOutput);

                neuralNetwork
                    .HiddenLayers
                    .Reverse()
                    .ToList()
                    .ForEach(SetHiddenLayerNeuronErrorRates);
            }
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

        public static BackPropagation For(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}
