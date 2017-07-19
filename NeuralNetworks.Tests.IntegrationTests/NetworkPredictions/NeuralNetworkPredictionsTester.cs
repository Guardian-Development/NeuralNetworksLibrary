using System;
using System.Linq; 
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;

namespace NeuralNetworks.Tests.IntegrationTests.NetworkPredictions
{
    public sealed class NeuralNetworkPredictionsTester
    {
        private NeuralNetworkContext context; 
        private IProvideRandomNumberGeneration randomGenerator; 
        private NeuralNetwork targetNeuralNetwork; 

        private NeuralNetworkPredictionsTester()
        {}

        public NeuralNetworkPredictionsTester NeuralNetworkEnvironment(
            NeuralNetworkContext context, 
            IProvideRandomNumberGeneration randomGenerator)
        {
            this.context = context; 
            this.randomGenerator = randomGenerator;
            return this; 
        }

        public NeuralNetworkPredictionsTester TargetNeuralNetwork(Action<ExplicitNeuralNetworkBuilder> action)
        {
            var neuralNetworkBuilder = ExplicitNeuralNetworkBuilder.CreateForTest(context, randomGenerator); 
            action.Invoke(neuralNetworkBuilder); 
            targetNeuralNetwork = neuralNetworkBuilder.Build(); 
            return this; 
        }

        public void InputAndExpectOutput(double[] inputs, double[] expectedOutput)
        {
            var expectedOutputAssertor = ExpectedOutputAssertorFor(expectedOutput); 
            var networkOutput = targetNeuralNetwork.PredictionFor(inputs); 
            expectedOutputAssertor.Assert(networkOutput); 
        }

        private IAssert<double[]> ExpectedOutputAssertorFor(double[] expectedOutput)
        {
            var expectedOutputAssertor = new OrderedListAssertor<double>();
            foreach (var output in expectedOutput)
            {
                var assertor = new EqualityAssertor<double>(output);
                expectedOutputAssertor.Assertors.Add(assertor); 
            }
            return expectedOutputAssertor; 
        }

        public static NeuralNetworkPredictionsTester Create()
            => new NeuralNetworkPredictionsTester(); 
    }
}