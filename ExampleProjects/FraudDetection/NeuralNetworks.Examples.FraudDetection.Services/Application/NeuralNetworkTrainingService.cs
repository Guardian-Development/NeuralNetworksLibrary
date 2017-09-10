using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;

namespace NeuralNetworks.Examples.FraudDetection.Services.Application
{
    public class NeuralNetworkTrainingService
    {
        private readonly NeuralNetworkAccessor networkAccessor;
        private readonly DataProvider dataProvider;

        public NeuralNetworkTrainingService(
            NeuralNetworkAccessor networkAccessor,
            DataProvider dataProvider)
        {
            this.networkAccessor = networkAccessor;
            this.dataProvider = dataProvider;
        }

        public async Task TrainConfiguredNetworkForEpochs(
            int epochs,
            NeuralNetworkTrainingConfiguration trainingConfig)
        {
            var trainingData = dataProvider.TrainingData
                .Select(transaction => transaction.ToTrainingData())
                .ToList();
            
            await TrainingController.For(BackPropagation.WithConfiguration(
                                            networkAccessor.TargetNetwork,
                                            ParallelOptionsExtensions.MultiThreadedOptions(trainingConfig.ThreadCount), 
                                            trainingConfig.LearningRate,
                                            trainingConfig.Momentum))
                                    .TrainForEpochsOrErrorThresholdMet(trainingData, epochs, trainingConfig.MinimumErrorThreshold);
        }
    }
}