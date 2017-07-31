using System;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Components;

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
            SetNeuronErrorGradients(targets);
            PropagateResultOfNeuronErrors();
            return CalculateError(targets);
        }

        private void SetNeuronErrorGradients(double[] targets)
        {
            neuralNetwork.OutputLayer.Neurons
                         .ForEach((neuron, i) => 
                            neuronErrorGradientCalculator.SetNeuronErrorGradient(neuron, targets[i]));

            neuralNetwork.HiddenLayers
                         .ApplyInReverse(layer => 
                            layer.Neurons.ForEach(neuronErrorGradientCalculator.SetNeuronErrorGradient));
        }

        private void PropagateResultOfNeuronErrors()
        {
            neuralNetwork.OutputLayer.Neurons
                .Where(NeuronNotProducingCorrectResult)
                .ForEach(synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights);

            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer =>
                    layer.Neurons
                        .Where(NeuronNotProducingCorrectResult)
                        .ForEach(synapseWeightCalculator.CalculateAndUpdateInputSynapseWeights));
        }

        private bool NeuronNotProducingCorrectResult(Neuron neuron)
            => neuron.ErrorGradient != 0; 

        private double CalculateError(params double[] targets)
        {
            var i = 0;
            return neuralNetwork.OutputLayer.Neurons.Sum(
                neuron => Math.Abs(
                    neuronErrorGradientCalculator.CalculateErrorForOutputAgainstTarget(neuron, targets[i++])));
        }

        public static BackPropagation WithConfiguration(
            NeuralNetwork network, 
            double learningRate = 1, 
            double momentum = 0)
            => new BackPropagation(network, learningRate, momentum);
    }
}