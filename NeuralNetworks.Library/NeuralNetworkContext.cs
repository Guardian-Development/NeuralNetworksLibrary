namespace NeuralNetworks.Library
{
    public sealed class NeuralNetworkContext
    {
        public int ErrorGradientDecimalPlaces { get; }
        public int OutputDecimalPlaces { get; }
        public int SynapseWeightDecimalPlaces { get; }

        public NeuralNetworkContext(
            int errorGradientDecimalPlaces = 15,
            int outputDecimalPlaces = 15,
            int synapseWeightDecimalPlaces = 15)
        {
            ErrorGradientDecimalPlaces = errorGradientDecimalPlaces;
            OutputDecimalPlaces = outputDecimalPlaces;
            SynapseWeightDecimalPlaces = synapseWeightDecimalPlaces; 
        }

        public static NeuralNetworkContext MaximumPrecision => new NeuralNetworkContext(); 
    }
}
