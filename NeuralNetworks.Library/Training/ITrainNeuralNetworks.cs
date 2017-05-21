namespace NeuralNetworks.Library.Training
{
    public interface ITrainNeuralNetworks
    {
        void TrainNetwork(
            double[][] trainingInputs,
            double[][] expectedOutputs,
            double errorThreshold = 0.0001);
    }
}
