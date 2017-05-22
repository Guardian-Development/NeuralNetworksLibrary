using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class HiddenLayer : Layer
    {

        public HiddenLayer(int neuronCount, ActivationType activationType)
            : base(neuronCount, activationType)
        {}

        public static HiddenLayer For(int neuronCount, ActivationType activationType)
        {
            return new HiddenLayer(neuronCount, activationType);
        }

        public override Layer NextLayer { get; set; }
    }
}
