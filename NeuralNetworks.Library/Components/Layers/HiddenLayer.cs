using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Library.Components.Layers
{
    public sealed class HiddenLayer : Layer
    {
        public HiddenLayer(int neuronCount, ActivationType activationType, Neuron biasNeuron)
            : base(neuronCount, activationType, biasNeuron)
        {}

        public static HiddenLayer For(
            int neuronCount,
            ActivationType activationType,
            double biasNeuronOutput = 1)
        {
            var biasNeuron = Neuron.For(activationType);
            biasNeuron.Output = biasNeuronOutput;
        
            return new HiddenLayer(neuronCount, activationType, biasNeuron);
        }

        public override Layer NextLayer { get; set; }
    }
}
