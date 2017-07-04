using System; 
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class LayerAssertor : IAssert<Layer>
    {
        public IAssert<IEnumerable<Neuron>> NeuronAssertors { get; set; }
            = FieldAssertor<IEnumerable<Neuron>>.NoAssert; 

        public void Assert(Layer actualItem)
        {
            NeuronAssertors.Assert(actualItem.Neurons);
        }

        public class Builder : IAssertBuilder<Layer>
        {
            private LayerAssertor assertor = new LayerAssertor();

            public Builder Neurons(params Action<NeuronAssertor.Builder>[] neuronAssertors)
            {
                //TODO: build up neurons assertors. 
                //TODO: implement neural network assertor
            }

            public IAssert<Layer> Build() => assertor;

            private static UnorderedListAssertor<int, Neuron> ListAssertorFor(
                Action<NeuronAssertor.Builder>[] synapseAssertors)
			{
                var listAssertor = new UnorderedListAssertor<int, Neuron>(GetKeyForAssertor);
				PopulateListAssertor(synapseAssertors, listAssertor);
				return listAssertor;
			}

            private static int GetKeyForAssertor(Neuron neuronToAssert)
                => neuronToAssert.Id; 

			private static void PopulateListAssertor(
                Action<NeuronAssertor.Builder>[] neuronAssertors,
                UnorderedListAssertor<int, Neuron> listAssertor)
			{
                foreach (Action<NeuronAssertor.Builder> produceAssertor in neuronAssertors)
				{
                    var builder = new NeuronAssertor.Builder();
					produceAssertor(builder);

                    var neuronAssertor = builder.Build() as NeuronAssertor;

					listAssertor.Assertors.Add(neuronAssertor.NeuronId, neuronAssertor);
				}
			}
        }
    }
}
