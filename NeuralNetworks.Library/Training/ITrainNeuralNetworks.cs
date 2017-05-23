namespace NeuralNetworks.Library.Training
{
    public interface ITrainNeuralNetworks
    {
        void TrainNetwork(
            double[][] trainingInputs,
            double[][] expectedOutputs,
            int epochs = 100,
            double errorThreshold = 0.0001);
    }
}
