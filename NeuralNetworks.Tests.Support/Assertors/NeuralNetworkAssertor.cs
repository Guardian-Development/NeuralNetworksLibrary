using System;
using System.Collections.Generic;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components;
using System.Linq; 

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class NeuralNetworkAssertor : IAssert<NeuralNetwork>
    {
        public IAssert<InputLayer> InputLayerAssertor { get; set; }
            = FieldAssertor<InputLayer>.NoAssert;

        public IAssert<IEnumerable<HiddenLayer>> HiddenLayersAssertor { get; set; }
            = FieldAssertor<IEnumerable<HiddenLayer>>.NoAssert;

        public IAssert<OutputLayer> OutputLayerAssertor { get; set; }
            = FieldAssertor<OutputLayer>.NoAssert; 

        public void Assert(NeuralNetwork actualItem)
        {
            InputLayerAssertor.Assert(actualItem.InputLayer);
            HiddenLayersAssertor.Assert(actualItem.HiddenLayers);
            OutputLayerAssertor.Assert(actualItem.OutputLayer);
        }

        public class Builder : IAssertBuilder<NeuralNetwork>
        {
            private readonly NeuralNetworkAssertor assertor = new NeuralNetworkAssertor();

            public Builder InputLayer(Action<LayerAssertor<InputLayer>.Builder> actions)
            {
                var layerAssertor = new LayerAssertor<InputLayer>.Builder(); 
                actions.Invoke(layerAssertor);
                assertor.InputLayerAssertor = layerAssertor.Build();
                return this; 
            }

            public Builder HiddenLayers(params Action<LayerAssertor<HiddenLayer>.Builder>[] actions)
            {
                var listAssertor = ListAssertorFor(actions);
                assertor.HiddenLayersAssertor = listAssertor;
                return this; 
            }

            public Builder OutputLayer(Action<LayerAssertor<OutputLayer>.Builder> actions)
            {
                var layerAssertor = new LayerAssertor<OutputLayer>.Builder();
				actions.Invoke(layerAssertor);
                assertor.OutputLayerAssertor = layerAssertor.Build();
				return this;
			}

            public IAssert<NeuralNetwork> Build() => assertor;

            private static UnorderedListAssertor<string, HiddenLayer> ListAssertorFor(
	            Action<LayerAssertor<HiddenLayer>.Builder>[] synapseAssertors)
			{
                var listAssertor = new UnorderedListAssertor<string, HiddenLayer>(GetKeyForAssertor);
				PopulateListAssertor(synapseAssertors, listAssertor);
				return listAssertor;
			}

            private static string GetKeyForAssertor(HiddenLayer layerToAssert)
                => CreateSynapseKey(layerToAssert.Neurons.Select(l => l.Id).ToArray());

			private static string CreateSynapseKey(int[] neuronIds)
                => string.Join(",", neuronIds);

			private static void PopulateListAssertor(
				Action<LayerAssertor<HiddenLayer>.Builder>[] synapseAssertors,
                UnorderedListAssertor<string, HiddenLayer> listAssertor)
			{
				foreach (var produceAssertor in synapseAssertors)
				{
					var builder = new LayerAssertor<HiddenLayer>.Builder();
					produceAssertor(builder);

					var layerAssertor = builder.Build() as LayerAssertor<HiddenLayer>;

					listAssertor.Assertors.Add(
                        CreateSynapseKey(layerAssertor.NeuronIds.ToArray()),
                        layerAssertor);
				}
			}
        }
    }
}
