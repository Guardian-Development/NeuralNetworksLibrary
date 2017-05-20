using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class HiddenLayer : Layer
    {
        private readonly Layer previousLayer;

        public HiddenLayer(int neuronCount, ActivationType activationType, Layer previousLayer)
            : base(neuronCount, activationType)
        {
            this.previousLayer = previousLayer;
        }

        public static HiddenLayer For(int neuronCount, ActivationType activationType, Layer previousLayer)
        {
            return new HiddenLayer(neuronCount, activationType, previousLayer);
        }
    }
}
