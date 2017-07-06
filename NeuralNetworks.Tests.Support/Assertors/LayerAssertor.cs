using System; 
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class LayerAssertor<TLayer> : IAssert<TLayer>
        where TLayer : Layer
    {
        public List<int> NeuronIds = new List<int>(); 

        public IAssert<IEnumerable<Neuron>> NeuronAssertors { get; set; }
            = FieldAssertor<IEnumerable<Neuron>>.NoAssert; 

        public void Assert(TLayer actualItem)
        {
            NeuronAssertors.Assert(actualItem.Neurons);
        }

        public class Builder : IAssertBuilder<TLayer>
        {
            private readonly LayerAssertor<TLayer> assertor = new LayerAssertor<TLayer>();

            public Builder Neurons(params Action<NeuronAssertor.Builder>[] neuronAssertors)
            {
                var listAssertor = ListAssertorFor(neuronAssertors);
                assertor.NeuronAssertors = listAssertor;
                return this; 
            }

            public IAssert<TLayer> Build() => assertor;

            private UnorderedListAssertor<int, Neuron> ListAssertorFor(
                Action<NeuronAssertor.Builder>[] neuronAssertors)
			{
                var listAssertor = new UnorderedListAssertor<int, Neuron>(GetKeyForAssertor);
				PopulateListAssertor(neuronAssertors, listAssertor);
				return listAssertor;
			}

            private static int GetKeyForAssertor(Neuron neuronToAssert)
                => neuronToAssert.Id; 

			private void PopulateListAssertor(
                Action<NeuronAssertor.Builder>[] neuronAssertors,
                UnorderedListAssertor<int, Neuron> listAssertor)
			{
                foreach (var produceAssertor in neuronAssertors)
				{
                    var builder = new NeuronAssertor.Builder();
					produceAssertor(builder);

                    var neuronAssertor = builder.Build() as NeuronAssertor; 
                    assertor.NeuronIds.Add(neuronAssertor.NeuronId);

					listAssertor.Assertors.Add(neuronAssertor.NeuronId, neuronAssertor);
				}
			}
        }
    }
}
