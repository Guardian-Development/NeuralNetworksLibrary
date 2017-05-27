using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training
{
    public sealed class BackPropagation
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
        }

        public void Train(List<TrainingDataSet> dataSets, int numEpochs)
        {
            for (var i = 0; i < numEpochs; i++)
            {
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Inputs);
                    BackPropagate(dataSet.Outputs);
                }
            }
        }

        public void Train(List<TrainingDataSet> dataSets, double minimumError)
        {
            var error = 1.0;
            var numEpochs = 0;

            while (error > minimumError && numEpochs < int.MaxValue)
            {
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Inputs);
                    BackPropagate(dataSet.Outputs);
                    errors.Add(CalculateError(dataSet.Outputs));
                }
                error = errors.Average();
                Log.LogInformation($"Error Rate: {error}. Epoch: {numEpochs}");

                numEpochs++;
            }
        }

        private void ForwardPropagate(params double[] inputs)
        {
            var i = 0;
            neuralNetwork.InputLayer.Neurons.ForEach(a => a.Value = inputs[i++]);
            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(a => a.CalculateOutput()));
            neuralNetwork.OutputLayer.Neurons.ForEach(a => a.CalculateOutput());
        }

        private void BackPropagate(params double[] targets)
        {
            var i = 0;
            neuralNetwork.OutputLayer.Neurons.ForEach(a => a.CalculateGradient(targets[i++]));
            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(a => a.CalculateGradient()));

            neuralNetwork.OutputLayer.Neurons.ForEach(a => a.UpdateWeights(learningRate, momentum));
            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(a => a.UpdateWeights(learningRate, momentum)));
        }

        public double[] Compute(params double[] inputs)
        {
            ForwardPropagate(inputs);
            return neuralNetwork.OutputLayer.Neurons.Select(a => a.Value).ToArray();
        }

        private double CalculateError(params double[] targets)
        {
            var i = 0;
            return neuralNetwork.OutputLayer.Neurons.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
        }

        public static BackPropagation For(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}