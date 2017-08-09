using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Library.Training
{
    public abstract class NeuralNetworkTrainer : ITrainNeuralNetworks
    {
        public NeuralNetwork NetworkUnderTraining { get; }

        protected NeuralNetworkTrainer(NeuralNetwork neuralNetworkUnderTraining)
        {
            NetworkUnderTraining = neuralNetworkUnderTraining; 
        }

        public abstract double PerformSingleEpochProducingErrorRate(TrainingDataSet trainingDataSet); 
    }
}