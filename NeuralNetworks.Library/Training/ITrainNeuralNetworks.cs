using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Library.Training
{
    public interface ITrainNeuralNetworks
    {
        double PerformSingleEpochProducingErrorRate(TrainingDataSet trainingDataSet);
    }
}
