using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training
{
    public sealed class TrainingController<TNeuralNetworkTrainer>
        where TNeuralNetworkTrainer : ITrainNeuralNetworks
    {
        private static ILogger Log => LoggerProvider.For<TrainingController<TNeuralNetworkTrainer>>();
        private readonly TNeuralNetworkTrainer neuralNetworkTrainer;

        internal TrainingController(TNeuralNetworkTrainer neuralNetworkTrainer)
        {
            this.neuralNetworkTrainer = neuralNetworkTrainer;
        }

        public async Task TrainForEpochsOrErrorThresholdMet(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs,
            double errorThreshold,
            CancellationToken cancellationToken = default(CancellationToken))
        {
            await Task.Run(() => {
                var error = 1.0;
                var numEpochs = 0;

                while (error > errorThreshold && numEpochs < maximumEpochs)
                {
                    var errors = trainingDataSet
                        .Select(neuralNetworkTrainer.PerformSingleEpochProducingErrorRate)
                        .ToList();

                    error = errors.Average();
                    Log.LogDebug($"Error Rate: {error}. Epoch: {numEpochs}");

                    numEpochs++;
                }
            }, cancellationToken); 
        }

        public async Task TrainForEpochs(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs,
            CancellationToken cancellationToken = default(CancellationToken))
        {
            await Task.Run(() => {
                var numEpochs = 0;

                while (numEpochs < maximumEpochs)
                {
                    foreach (var dataSet in trainingDataSet)
                    {
                        neuralNetworkTrainer.PerformSingleEpochProducingErrorRate(dataSet);
                    }

                    Log.LogDebug($"Epochs performed: {numEpochs}");

                    numEpochs++;
                }
            }, cancellationToken);
        }

        public async Task TrainForErrorThreshold(
            IList<TrainingDataSet> trainingDataSet,
            double minimumErrorThreshold,
            CancellationToken cancellationToken = default(CancellationToken))
        {
            await Task.Run(() => {
                var error = minimumErrorThreshold + 1;

                while (error > minimumErrorThreshold)
                {
                    var errors = trainingDataSet
                        .Select(neuralNetworkTrainer.PerformSingleEpochProducingErrorRate)
                        .ToList();

                    error = errors.Average();
                    Log.LogDebug($"Current Error Rate: {error}");
                }
            }, cancellationToken); 
        }
    }

    public static class TrainingController
    {
       public static TrainingController<TNeuralNetworkTrainer> For<TNeuralNetworkTrainer>(TNeuralNetworkTrainer trainer)
            where TNeuralNetworkTrainer : ITrainNeuralNetworks
        {
            return new TrainingController<TNeuralNetworkTrainer>(trainer);
        }
    }
}