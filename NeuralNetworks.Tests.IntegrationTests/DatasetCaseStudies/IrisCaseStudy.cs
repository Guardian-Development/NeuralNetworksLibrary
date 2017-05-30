using System.Reflection;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies
{
    public sealed class IrisCaseStudy : NeuralNetworkTest
    {
        [Fact]
        public void CanSuccessfullySolveIrisProblem()
        {
            var assembly = typeof(IrisCaseStudy).GetTypeInfo().Assembly;

            var irisData = ReadCsv.FromEmbeddedResource(assembly,
                "NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.Iris.csv",
                IrisDataRow.For);
        }
    }
}