using System;

namespace NeuralNetworks.Library.Training
{
    public sealed class BackPropagation : ITrainNeuralNetworks
    {
        private readonly NeuralNetwork neuralNetwork;
        private readonly double learningRate;
        private readonly double momentum; 

        private BackPropagation(NeuralNetwork neuralNetwork, double learningRate, double momentum)
        {
            this.neuralNetwork = neuralNetwork;
            this.learningRate = learningRate;
            this.momentum = momentum; 
        }

        public void TrainNetwork(
            double[][] trainingInputs, 
            double[][] expectedOutputs, 
            double errorThreshold = 0.0001)
        {
            ValidateTrainingInputsWithExepctedOutputs(trainingInputs, expectedOutputs);

            for (var i = 0; i < trainingInputs.Length; i++)
            {
                MatchTrainingInputToExpectedOutput(i, 
                    trainingInputs, 
                    expectedOutputs, 
                    out var inputRow, 
                    out var expectedOutputRow);
;
                var predictedOutput = neuralNetwork.MakePrediction(inputRow);
                //now need to proagate error rate back through the network. 
            }
        }

        private void ValidateTrainingInputsWithExepctedOutputs(
            double[][] trainingInputs,
            double[][] expectedOutputs)
        {
            if (trainingInputs.Length != expectedOutputs.Length)
            {
                throw new ArgumentException(
                    $"{nameof(trainingInputs)} must be the same length as {expectedOutputs}");
            }
        }

        private void MatchTrainingInputToExpectedOutput(
            int index,
            double[][] trainingInputs,
            double[][] correspondingOutputs,
            out double[] inputForIndex,
            out double[] outputForIndex)
        {
            inputForIndex = trainingInputs[index];
            outputForIndex = correspondingOutputs[index]; 
        }

        public static BackPropagation For(NeuralNetwork network, double learningRate, double momentum)
        {
            return new BackPropagation(network, learningRate, momentum);
        }
    }
}
