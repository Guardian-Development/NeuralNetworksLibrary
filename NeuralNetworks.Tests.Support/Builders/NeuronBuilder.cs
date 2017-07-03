using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Tests.Support.Builders
{
    public abstract class NeuronBuilderBase<TBuilder>
        where TBuilder : NeuronBuilderBase<TBuilder>
    {
        protected double ErrorValue;
        protected double OutputValue;
        protected ActivationType ActivationType; 

        public TBuilder ErrorRate(double error)
        {
            ErrorValue = error;
            return (TBuilder) this;
        }

        public TBuilder Output(double output)
        {
            OutputValue = output;
            return (TBuilder) this;
        }

        public TBuilder Activation(ActivationType type)
        {
            ActivationType = type;
            return (TBuilder) this; 
        }
    }

    public sealed class NeuronBuilder : NeuronBuilderBase<NeuronBuilder>
    {
        public Neuron Build(NeuralNetworkContext context)
        {
            var neuron = Neuron.For(context, ActivationType);
            neuron.ErrorRate = ErrorValue;
            neuron.Output = OutputValue;
            return neuron; 
        }
    }

    public sealed class BiasNeuronBuilder : NeuronBuilderBase<BiasNeuronBuilder>
    {
        public BiasNeuron Build(NeuralNetworkContext context)
        {
            var neuron = BiasNeuron.For(context, ActivationType, OutputValue);
            neuron.ErrorRate = ErrorValue;
            return neuron; 
        }
    }
}