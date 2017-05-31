using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Data;
using NeuralNetworks.Library.Training;
using NeuralNetworks.Library.Training.BackPropagation;
using NeuralNetworks.Tests.Support;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies
{
    public sealed class IrisCaseStudy : NeuralNetworkTest
    {
        [Fact]
        public void CanSuccessfullySolveIrisProblem()
        {
            var neuralNetwork = NeuralNetwork.For()
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.Sigmoid)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.Sigmoid)
                .WithOutputLayer(neuronCount: 1, activationType: ActivationType.Sigmoid)
                .Build();

            TrainingController<BackPropagation>
                .For(BackPropagation.WithConfiguration(neuralNetwork, learningRate: 0.9, momentum: 0.9))
                .TrainForErrorThreshold(GetIrisTrainingData(), minimumErrorThreshold: 0.4);

            GetIrisTestData().ForEach(testDataRow =>
            {
                var inputs = new[]
                {
                    testDataRow.PetalLengthCm,
                    testDataRow.PetalWidthCm,
                    testDataRow.SepalLengthCm,
                    testDataRow.SepalWidthCm
                };

                var prediction = neuralNetwork.PredictionFor(inputs);
            });
        }

        private static List<TrainingDataSet> GetIrisTrainingData()
        {
            var assembly = typeof(IrisCaseStudy).GetTypeInfo().Assembly;

            var irisData = ReadCsv.FromEmbeddedResource(assembly,
                "NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.Iris.csv",
                IrisDataRow.For).ToList();

            return DataForFilter(irisData, filter: (_, i) => i % 2 != 0);
        }

        private static List<TrainingDataSet> DataForFilter(
            IEnumerable<IrisDataRow> irisData,
            Func<IrisDataRow, int, bool> filter)
        {
            return irisData.Where(filter)
                .Select(datarow => new
                {
                    inputs = new[]
                    {
                        datarow.PetalLengthCm,
                        datarow.PetalWidthCm,
                        datarow.SepalLengthCm,
                        datarow.SepalWidthCm
                    },
                    output = new[] { datarow.Species }
                })
                .Select(arrayConversion =>
                    TrainingDataSet.For(arrayConversion.inputs, arrayConversion.output))
                .ToList();
        }

        private static List<IrisDataRow> GetIrisTestData()
        {
            var assembly = typeof(IrisCaseStudy).GetTypeInfo().Assembly;

            var irisData = ReadCsv.FromEmbeddedResource(assembly,
                "NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.Iris.csv",
                IrisDataRow.For).ToList();

            return irisData.Where((dataRow, i) => i % 2 == 0).ToList(); 
        }
    }
}