using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.Support.Builders
{
    public abstract class LayerBuilder<TBuilder>
        where TBuilder : LayerBuilder<TBuilder>
    {
        protected readonly List<(int id, Neuron neuron)> NeuronsWithId = new List<(int id, Neuron neuron)>();
        protected (int id, BiasNeuron neuron) BiasNeuronWithId;
        private readonly NeuralNetworkContext context;

        protected LayerBuilder(NeuralNetworkContext context)
        {
            this.context = context;
        }

        public IEnumerable<(int id, Neuron neuron)> AllNeurons
            => BiasNeuronWithId.neuron == null ? NeuronsWithId : NeuronsWithId.Append(BiasNeuronWithId);

        public TBuilder Neuron(int id, Action<NeuronBuilder> actions)
        {
            var builder = new NeuronBuilder();
            actions.Invoke(builder);
            var neuron = builder.Build(context);
            NeuronsWithId.Add(ValueTuple.Create(id, neuron));
            return (TBuilder) this;
        }

        public virtual TBuilder BiasNeuron(int id, Action<BiasNeuronBuilder> actions)
        {
            var builder = new BiasNeuronBuilder();
            actions.Invoke(builder);
            var biasNeuron = builder.Build(context);
            BiasNeuronWithId = ValueTuple.Create(id, biasNeuron);
            return (TBuilder) this;
        }
    }

    public sealed class InputLayerBuilder : LayerBuilder<InputLayerBuilder>
    {
        public InputLayerBuilder(NeuralNetworkContext context) 
            : base(context)
        {}

        public InputLayer Build() =>
            InputLayer.For(
                NeuronsWithId.Select(neuron => neuron.neuron).ToList(),
                BiasNeuronWithId.neuron);
    }

    public sealed class HiddenLayerBuilder : LayerBuilder<HiddenLayerBuilder>
    {
        public HiddenLayerBuilder(NeuralNetworkContext context) 
            : base(context)
        {}

        public HiddenLayer Build() =>
            HiddenLayer.For(
                NeuronsWithId.Select(neuron => neuron.neuron).ToList(),
                BiasNeuronWithId.neuron);
    }

    public sealed class OutputLayerBuilder : LayerBuilder<OutputLayerBuilder>
    {
        public OutputLayerBuilder(NeuralNetworkContext context) : base(context)
        {}

        public override OutputLayerBuilder BiasNeuron(int id, Action<BiasNeuronBuilder> actions)
            => throw new InvalidOperationException("Output layer not allowed bias neuron.");

        public OutputLayer Build() =>
            OutputLayer.For(NeuronsWithId.Select(neuron => neuron.neuron).ToList());
    }
}