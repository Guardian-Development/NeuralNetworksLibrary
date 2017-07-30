using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class SynapseAssertor : IAssert<Synapse>
    {
        public int? InputNeuronId { get; private set; }
        public int? OutputNeuronId { get; private set; }

        public IAssert<int> InputNeuronIdAssertor{ get; set; }
            = FieldAssertor<int>.NoAssert;

        public IAssert<int> OutputNeuronIdAssertor { get; set; }
            = FieldAssertor<int>.NoAssert;

		public IAssert<double> WeightAssertor { get; set; }
            = FieldAssertor<double>.NoAssert;

		public IAssert<double> WeightDeltaAssertor { get; set; }
            = FieldAssertor<double>.NoAssert;

		public void Assert(Synapse actualItem)
        {
            InputNeuronIdAssertor.Assert(actualItem.InputNeuron.Id);
            OutputNeuronIdAssertor.Assert(actualItem.OutputNeuron.Id);

            WeightAssertor.Assert(actualItem.Weight);
            WeightDeltaAssertor.Assert(actualItem.WeightDelta);
        }

        public class Builder : IAssertBuilder<Synapse>
        {
            private readonly SynapseAssertor assertor = new SynapseAssertor();

			public Builder InputNeuronId(int id)
			{
				assertor.InputNeuronIdAssertor = new EqualityAssertor<int>(id);
				assertor.InputNeuronId = id;
				return this;
			}

            public Builder OutputNeuronId(int id)
            {
                assertor.OutputNeuronIdAssertor = new EqualityAssertor<int>(id);
                assertor.OutputNeuronId = id;
				return this;
            }

            public Builder Weight(double expectedWeight)
            {
                assertor.WeightAssertor = new EqualityAssertor<double>(expectedWeight);
                return this; 
            }

            public Builder WeightDelta(double expectedWeightDelta, int precisionToAssertTo)
            {
                assertor.WeightDeltaAssertor = new RoundedDoubleAssertor(expectedWeightDelta, precisionToAssertTo); 
                return this; 
            }

			public IAssert<Synapse> Build() => assertor;
		}
    }
}
