using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class InputLayer : Layer
    {
        public InputLayer(int neuronCount, ActivationType activationType, Neuron biasNeuron)
            : base(neuronCount, activationType, biasNeuron)
        {}

        public static InputLayer For(
            int neuronCount, 
            ActivationType activationType, 
            double biasNeuronOutput = 1)
        {
            var biasNeuron = Neuron.For(activationType);
            biasNeuron.Output = biasNeuronOutput; 

            return new InputLayer(neuronCount, activationType, biasNeuron);
        }

        public override Layer NextLayer { get; set; }
    }
}
