namespace NeuralNetworks.Examples.FraudDetection.Services.Configuration
{
    public class NeuralNetworkTrainingConfiguration 
    {
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public int ThreadCount { get; set; }
        public double MinimumErrorThreshold { get; set; }
    }
}