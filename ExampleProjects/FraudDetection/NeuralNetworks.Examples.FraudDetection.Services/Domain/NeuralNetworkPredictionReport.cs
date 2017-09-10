namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public sealed class NeuralNetworkPredictionsReport
    {
        public int NumberOfCorrectPredictions { get; }
        public int NumberOfIncorrectPredictions { get; }

        internal NeuralNetworkPredictionsReport(
            int numberOfCorrectPredictions, 
            int numberOfIncorrectPredictions)
        {
            NumberOfCorrectPredictions = numberOfCorrectPredictions; 
            NumberOfIncorrectPredictions = numberOfIncorrectPredictions; 
        }
    }
}