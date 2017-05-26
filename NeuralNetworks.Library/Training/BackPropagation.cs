using System;
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
                    ForwardPropagate(dataSet.Inputs);
                    BackPropagate(dataSet.Outputs);
                    errors.Add(CalculateError(dataSet.Outputs));
                }
                error = errors.Average();
                numEpochs++;
            }
        }

        private void ForwardPropagate(double[] inputs)
        {
            throw new NotImplementedException();
        }

        private void BackPropagate(double[] expectedOutputs)
        {
            throw new NotImplementedException();
        }

        private double CalculateError(double[] targets)
        {
            throw new NotImplementedException();
        }

        public static BackPropagation For(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}