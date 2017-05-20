using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class InputLayer : Layer
    {
        public InputLayer(int neuronCount, ActivationType activationType)
            : base(neuronCount, activationType)
        { }

        public static InputLayer For(int neuronCount, ActivationType activationType)
        {
            return new InputLayer(neuronCount, activationType);
        }
    }
}
