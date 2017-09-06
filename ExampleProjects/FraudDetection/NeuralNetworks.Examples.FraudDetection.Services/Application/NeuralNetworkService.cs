using System.Threading.Tasks;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Examples.FraudDetection.Services.Application
{
    public class NeuralNetworkService
    {
        private readonly NeuralNetworkConfiguration neuralNetworkConfiguration;
        private readonly IProvideNeuralNetworkTrainingData networkTrainingDataProvider;

        public NeuralNetworkService(
            NeuralNetworkConfiguration neuralNetworkConfiguration,
            IProvideNeuralNetworkTrainingData networkTrainingDataProvider)
        {
            this.neuralNetworkConfiguration = neuralNetworkConfiguration;
            this.networkTrainingDataProvider = networkTrainingDataProvider;
        }

        public async Task BeginTraining(int epochAmount)
        {
            await neuralNetworkConfiguration.NeuralNetworkTrainer
                                            .TrainForEpochs(networkTrainingDataProvider.TrainingData, epochAmount); 
        }
    }
}