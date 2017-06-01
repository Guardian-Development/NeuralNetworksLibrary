using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.Logging;
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
                .WithInputLayer(neuronCount: 4, activationType: ActivationType.Sigmoid, biasOutput: 0)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.TanH, biasOutput: 1)
                .WithHiddenLayer(neuronCount: 20, activationType: ActivationType.TanH, biasOutput: 0)
                .WithOutputLayer(neuronCount: 3, activationType: ActivationType.Sigmoid)
                .Build();

            TrainingController<BackPropagation>
                .For(BackPropagation.WithConfiguration(neuralNetwork, learningRate: 0.6, momentum: 0.4))
                .TrainForErrorThreshold(GetIrisTrainingData(), minimumErrorThreshold: 0.15);

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
                LogReultForPrediction(prediction, testDataRow.Species);
            });
        }
        
        private void LogReultForPrediction(double[] prediction, double actual)
        {
            var indexOfMax = -1;
            double currentMax = -1; 
            for (var i = 0; i < prediction.Length; i++)
            {
                if (currentMax > prediction[i])
                {
                   continue;
                }
                currentMax = prediction[i];
                indexOfMax = i;
            }

            LogFor(typeof(IrisCaseStudy)).LogInformation($"Predicted for {actual} : {indexOfMax + 1}");
        }

        private static List<TrainingDataSet> GetIrisTrainingData()
        {
            var assembly = typeof(IrisCaseStudy).GetTypeInfo().Assembly;
            var test = assembly.GetManifestResourceNames(); 

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
                    output = BuildOutputForSpecied(datarow.Species)
                })
                .Select(arrayConversion =>
                    TrainingDataSet.For(arrayConversion.inputs, arrayConversion.output))
                .ToList();
        }

        private static double[] BuildOutputForSpecied(double species)
        {
            switch (species)
            {
                case 1:
                    return new[] {1.0, 0.0, 0.0};
                case 2:
                    return new[] {0.0, 1.0, 0.0};
                case 3:
                    return new[] {0.0, 0.0, 1.0};
                default:
                    throw new ArgumentException("Unsupported species"); 
            }
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