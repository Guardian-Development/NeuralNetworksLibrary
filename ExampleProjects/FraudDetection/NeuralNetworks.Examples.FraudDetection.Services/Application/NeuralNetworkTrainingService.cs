using System.Collections.Generic;
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
        private readonly TrainingDataProvider trainingDataProvider;

        public NeuralNetworkTrainingService(
            NeuralNetworkAccessor networkAccessor,
            TrainingDataProvider trainingDataProvider)
        {
            this.networkAccessor = networkAccessor;
            this.trainingDataProvider = trainingDataProvider;
        }

        public async Task TrainConfiguredNetworkForEpochs(
            int epochs, 
            NeuralNetworkTrainingConfiguration trainingConfig)
        {
            await TrainingController.For(BackPropagation.WithConfiguration(
                                            networkAccessor.TargetNetwork,
                                            ParallelOptionsExtensions.MultiThreadedOptions(trainingConfig.ThreadCount), 
                                            trainingConfig.LearningRate,
                                            trainingConfig.Momentum))
                                    .TrainForEpochs(trainingDataProvider.TrainingData, epochs);
        }
    }
}