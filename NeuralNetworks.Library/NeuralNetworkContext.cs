namespace NeuralNetworks.Library
{
    public sealed class NeuralNetworkContext
    {
        public int ErrorRateDecimalPlaces { get; }
        public int OutputDecimalPlaces { get; }
        public int SynapseWeightDecimalPlaces { get; }

        public NeuralNetworkContext(
            int errorRateDecimalPlaces = 15,
            int outputDecimalPlaces = 15,
            int synapseWeightDecimalPlaces = 15)
        {
            ErrorRateDecimalPlaces = errorRateDecimalPlaces;
            OutputDecimalPlaces = outputDecimalPlaces;
            SynapseWeightDecimalPlaces = synapseWeightDecimalPlaces; 
        }

        public static NeuralNetworkContext MaximumPrecision => new NeuralNetworkContext(); 
    }
}
