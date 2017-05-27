﻿using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
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
        }

        public void TrainNetwork(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs = 100,
            double errorThreshold = 0.0001)
        {
            var error = 1.0;
            var numEpochs = 0;

            while (error > errorThreshold && numEpochs < maximumEpochs)
            {
                var errors = new List<double>();
                foreach (var dataSet in trainingDataSet)
                {
                    neuralNetwork.PredictionFor(dataSet.Inputs);
                    BackPropagate(dataSet.Outputs);
                    errors.Add(CalculateError(dataSet.Outputs));
                }
                error = errors.Average();
                Log.LogInformation($"Error Rate: {error}. Epoch: {numEpochs}");

                numEpochs++;
            }
        }

        private void BackPropagate(params double[] targets)
        {
            var i = 0;
            neuralNetwork.OutputLayer.Neurons.ForEach(a => a.CalculateErrorGradient(targets[i++]));
            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(a => a.CalculateErrorGradient()));

            neuralNetwork.OutputLayer.Neurons.ForEach(a => a.UpdateWeights(learningRate, momentum));
            neuralNetwork.HiddenLayers
                .ApplyInReverse(layer => layer.Neurons.ForEach(a => a.UpdateWeights(learningRate, momentum)));
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