using NeuralNetworks.Library;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public class NeuralNetworkAccessor 
    {
        internal NeuralNetwork TargetNetwork { get; }

        public NeuralNetworkAccessor(NeuralNetwork targetNetwork)
        {
            TargetNetwork = targetNetwork;
        }
    }
}