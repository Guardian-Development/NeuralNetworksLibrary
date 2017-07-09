using System;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.Extensions; 

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class BackPropagation : NeuralNetworkTrainer
    {
        private readonly NeuralNetwork neuralNetwork;
		private readonly NeuronErrorGradientCalculator neuronErrorGradientCalculator;
		private readonly SynapseWeightCalculator synapseWeightCalculator;

        private BackPropagation(NeuralNetwork neuralNetwork, double learningRate, double momentum)
            : base(neuralNetwork)
        {
            this.neuralNetwork = neuralNetwork;
            neuronErrorGradientCalculator = NeuronErrorGradientCalculator.Create(); 
            synapseWeightCalculator = SynapseWeightCalculator.For(learningRate, momentum);
        }

        public override double PerformSingleEpochProducingErrorRate(TrainingDataSet trainingDataSet)
        {
            neuralNetwork.PredictionFor(trainingDataSet.Inputs);
            return BackPropagate(trainingDataSet.Outputs);
        }

        private double BackPropagate(params double[] targets)
        {
            SetNeuronErrorRates(targets);
            PropagateResultOfNeuronErrors();
            return CalculateError(targets);
        }

        private void SetNeuronErrorRates(double[] targets)
        {
            var i = 0;
            neuralNetwork.OutputLayer.Neurons
                         .ForEach(a => neuronErrorGradientCalculator.SetNeuronErrorGradient(a, targets[i++]));

            neuralNetwork.HiddenLayers
                         .ApplyInReverse(layer => layer.Neurons.ForEach(neuronErrorGradientCalculator.SetNeuronErrorGradient));
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
                    neuronErrorGradientCalculator.CalculateErrorForOutputAgainstTarget(neuron, targets[i++])));
        }

        public static BackPropagation WithConfiguration(NeuralNetwork network, double learningRate, double momentum)
            => new BackPropagation(network, learningRate, momentum);
    }
}