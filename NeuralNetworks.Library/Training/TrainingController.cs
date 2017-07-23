using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training
{
    public class TrainingController<TNeuralNetworkTrainer>
        where TNeuralNetworkTrainer : ITrainNeuralNetworks
    {
        private static ILogger Log => LoggerProvider.For<TrainingController<TNeuralNetworkTrainer>>();
        private readonly TNeuralNetworkTrainer neuralNetworkTrainer;

        private TrainingController(TNeuralNetworkTrainer neuralNetworkTrainer)
        {
            this.neuralNetworkTrainer = neuralNetworkTrainer;
        }

        public void TrainForEpochsOrErrorThresholdMet(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs,
            double errorThreshold)
        {
            var error = 1.0;
            var numEpochs = 0;

            while (error > errorThreshold && numEpochs < maximumEpochs)
            {
                var errors = trainingDataSet
                    .Select(neuralNetworkTrainer.PerformSingleEpochProducingErrorRate)
                    .ToList();

                error = errors.Average();
                Log.LogInformation($"Error Rate: {error}. Epoch: {numEpochs}");

                numEpochs++;
            }
        }

        public void TrainForEpochs(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs)
        {
            var numEpochs = 0;

            while (numEpochs < maximumEpochs)
            {
                trainingDataSet.ForEach
                foreach (var dataSet in trainingDataSet)
                {
                    neuralNetworkTrainer.PerformSingleEpochProducingErrorRate(dataSet);
                }

                Log.LogInformation($"Epochs performed: {numEpochs}");

                numEpochs++;
            }
        }

        public void TrainForErrorThreshold(
            IList<TrainingDataSet> trainingDataSet,
            double minimumErrorThreshold)
        {
            var error = 1.0;

            while (error > minimumErrorThreshold)
            {
                var errors = trainingDataSet
                    .Select(neuralNetworkTrainer.PerformSingleEpochProducingErrorRate)
                    .ToList();

                error = errors.Average();
                Log.LogInformation($"Current Error Rate: {error}");
            }
        }

        public static TrainingController<TNeuralNetworkTrainer> For(TNeuralNetworkTrainer trainer)
            => new TrainingController<TNeuralNetworkTrainer>(trainer);
    }
}