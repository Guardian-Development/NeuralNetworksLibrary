using System;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Library;
using NeuralNetworks.Library.NetworkInitialisation;
using NeuralNetworks.Tests.Support;
using NeuralNetworks.Tests.Support.Assertors;
using NeuralNetworks.Tests.Support.Builders;

namespace NeuralNetworks.Tests.IntegrationTests.NetworkPredictions
{
    public sealed class NeuralNetworkPredictionsTester : NeuralNetworkTester<NeuralNetworkPredictionsTester>
    {
        private NeuralNetworkPredictionsTester()
        {}

        public NeuralNetworkPredictionsTester InputAndExpectOutput(
            double[] inputs, 
            double[] expectedOutput, 
            ParallelOptions parallelOptions)
        {
            var expectedOutputAssertor = ExpectedOutputAssertorFor(expectedOutput); 
            var networkOutput = targetNeuralNetwork.PredictionFor(inputs, parallelOptions); 
            expectedOutputAssertor.Assert(networkOutput); 
            
            return this; 
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