using System.Collections.Generic;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Library.Training
{
    public interface ITrainNeuralNetworks
    {
        void TrainNetwork(
            IList<TrainingDataSet> trainingDataSet,
            int maximumEpochs = 100,
            double errorThreshold = 0.0001);
    }
}
