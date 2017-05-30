using System;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class BackPropagation : ITrainNeuralNetworks
    {
        private static ILogger Log => LoggerProvider.For<BackPropagation>();

        private readonly NeuralNetwork neuralNetwork;
        private readonly SynapseWeightCalculator synapseWeightCalculator;

        private BackPropagation(NeuralNetwork neuralNetwork, double learningRate, double momentum)
        {
            this.neuralNetwork = neuralNetwork;
            synapseWeightCalculator = SynapseWeightCalculator.For(learningRate, momentum);
        }

        public double PerformSingleEpochProducingErrorRate(TrainingDataSet trainingDataSet)
        {
            neuralNetwork.PredictionFor(trainingDataSet.Inputs);
            return BackPropagate(trainingDataSet.Outputs);
        }

        private double BackPropagate(params double[] targets)
        {
            CalculateNeuronErrorRates(targets);
            PropagateResultOfNeuronErrors();
            return CalculateError(targets);
        }

        private void CalculateNeuronErrorRates(double[] targets)
        {
            var i = 0;
            neuralNetwork.OutputLayer.Neurons
                .ForEach(a => NeuronErrorGradientOperations.SetNeuronErrorGradient(a, targets[i++]));

            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(NeuronErrorGradientOperations.SetNeuronErrorGradient));
        }

        private void PropagateResultOfNeuronErrors()
        {
            neuralNetwork.OutputLayer.Neurons
                .ForEach(synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights);

            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer =>
                    layer.Neurons.ForEach(synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights));
        }

        private double CalculateError(params double[] targets)
        {
            var i = 0;
            return neuralNetwork.OutputLayer.Neurons.Sum(
                neuron => Math.Abs(
                    NeuronErrorGradientOperations.CalculateErrorForOutputAgainstTarget(neuron, targets[i++])));
        }

        public static BackPropagation WithConfiguration(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}