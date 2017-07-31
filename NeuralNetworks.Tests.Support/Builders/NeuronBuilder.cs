using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation;

namespace NeuralNetworks.Tests.Support.Builders
{
    public abstract class AbstractNeuronBuilder<TBuilder, TNeuron>
        where TBuilder : AbstractNeuronBuilder<TBuilder, TNeuron>
        where TNeuron : Neuron
    {
        protected int id;
        protected double errorGradient;
        protected double output;
        protected ActivationType activationType;

        protected readonly NeuralNetworkContext Context;

        protected AbstractNeuronBuilder(NeuralNetworkContext context)
        {
            Context = context;
        }

        public TBuilder Id(int id)
        {
            this.id = id;
            return (TBuilder)this; 
        }

        public TBuilder ErrorGradient(double errorGradient)
        {
            this.errorGradient = errorGradient;
            return (TBuilder)this; 
        }

        public TBuilder Output(double output)
        {
            this.output = output;
            return (TBuilder)this; 
        }

        public TBuilder Activation(ActivationType activationType)
        {
            this.activationType = activationType;
            return (TBuilder)this; 
        }

        public abstract TNeuron Build(); 
    }

    public sealed class NeuronBuilder : AbstractNeuronBuilder<NeuronBuilder, Neuron>
    {
        public NeuronBuilder(NeuralNetworkContext context) 
            : base(context)
        {}

        public override Neuron Build()
        {
            var neuron = Neuron.For(Context, activationType); 
			neuron.Id = id;
			neuron.Output = output;
			neuron.ErrorGradient = errorGradient;
			return neuron;
		}
    }

    public sealed class BiasNeuronBuilder : AbstractNeuronBuilder<BiasNeuronBuilder, BiasNeuron>
    {
        public BiasNeuronBuilder(NeuralNetworkContext context) 
            : base(context)
        {}

        public override BiasNeuron Build()
        {
            var neuron = BiasNeuron.For(Context, activationType, output);
            neuron.Id = id;
            neuron.ErrorGradient = errorGradient;
            return neuron; 
        }
    }
}
