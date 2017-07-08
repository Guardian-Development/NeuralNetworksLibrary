using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Components.Activation.Functions;
using System.Collections.Generic;
using System;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class NeuronAssertor : IAssert<Neuron>
    {
        public int NeuronId { get; private set; }

        public IAssert<int> NeuronIdAssertor { get; set; }
            = FieldAssertor<int>.NoAssert; 

        public IAssert<IProvideNeuronActivation> ActivationFunctionAssertor { get; set; } 
            = FieldAssertor<IProvideNeuronActivation>.NoAssert; 

        public IAssert<IEnumerable<Synapse>> InputSynapsesAssertor { get; set; }
            = FieldAssertor<IEnumerable<Synapse>>.NoAssert;

		public IAssert<IEnumerable<Synapse>> OutputSynapsesAssertor { get; set; }
            = FieldAssertor<IEnumerable<Synapse>>.NoAssert;

        public IAssert<double> ErrorRateAssertor { get; set; }
            = FieldAssertor<double>.NoAssert;

        public IAssert<double> OutputAssertor { get; set; }
            = FieldAssertor<double>.NoAssert;

        public void Assert(Neuron actualItem)
        {
            NeuronIdAssertor.Assert(actualItem.Id);

            ActivationFunctionAssertor.Assert(actualItem.ActivationFunction);
            ErrorRateAssertor.Assert(actualItem.ErrorRate);
            OutputAssertor.Assert(actualItem.Output);

            InputSynapsesAssertor.Assert(actualItem.InputSynapses);
            OutputSynapsesAssertor.Assert(actualItem.OutputSynapses);
        }

        public class Builder : IAssertBuilder<Neuron>
        {
            private readonly NeuronAssertor assertor = new NeuronAssertor(); 

            public Builder Id(int id)
            {
                assertor.NeuronIdAssertor = new EqualityAssertor<int>(id);
                assertor.NeuronId = id;

                return this; 
            }

            public Builder InputSynapses(params Action<SynapseAssertor.Builder>[] synapseAssertors)
            {
				var listAssertor = ListAssertorFor(synapseAssertors);
                assertor.InputSynapsesAssertor = listAssertor;
				return this;
            }

            public Builder OutputSynapses(params Action<SynapseAssertor.Builder>[] synapseAssertors)
            {
                var listAssertor = ListAssertorFor(synapseAssertors);
                assertor.OutputSynapsesAssertor = listAssertor;
                return this;
            }

            public Builder ErrorRate(double errorRate)
            {
                assertor.ErrorRateAssertor = new EqualityAssertor<double>(errorRate);
                return this; 
            }

            public Builder Output(double output)
            {
                assertor.OutputAssertor = new EqualityAssertor<double>(output);
                return this; 
            }

            public IAssert<Neuron> Build() => assertor;

            private UnorderedListAssertor<string, Synapse> ListAssertorFor(
                Action<SynapseAssertor.Builder>[] synapseAssertors)
            {
                var listAssertor = new UnorderedListAssertor<string, Synapse>(GetKeyForAssertor);
                PopulateListAssertor(synapseAssertors, listAssertor);
                return listAssertor;
            }

            private static string GetKeyForAssertor(Synapse synapseToAssert)
                => CreateSynapseKey(synapseToAssert.InputNeuron.Id, synapseToAssert.OutputNeuron.Id); 

            private static string CreateSynapseKey(int inputNeuronId, int outputNeuronId)
                => $"{inputNeuronId}:{outputNeuronId}";

            private void PopulateListAssertor(
                Action<SynapseAssertor.Builder>[] synapseAssertors, 
                UnorderedListAssertor<string, Synapse> listAssertor)
            {
                foreach (var produceAssertor in synapseAssertors)
                {
                    var builder = new SynapseAssertor.Builder(); 
                    produceAssertor(builder);

                    var synapseAssertor = builder.Build() as SynapseAssertor; 

                    var inputNeuronId = synapseAssertor.InputNeuronId.HasValue ? 
                        synapseAssertor.InputNeuronId.Value : assertor.NeuronId; 

                    var outputNeuronId = synapseAssertor.OutputNeuronId.HasValue ? 
                    synapseAssertor.OutputNeuronId.Value : assertor.NeuronId;

					listAssertor.Assertors.Add(
                        CreateSynapseKey(inputNeuronId, outputNeuronId),
                        synapseAssertor);
                }
            }
        }
    }
}
