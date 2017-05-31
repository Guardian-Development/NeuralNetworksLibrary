using System.Collections.Generic;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies
{
    public class IrisDataRow
    {
        private static readonly Dictionary<string, double> SpeciesCategoriesToDouble 
            = new Dictionary<string, double>
        {
            ["Iris-setosa"] = 1.0,
            ["Iris-versicolor"] = 2.0,
            ["Iris-virginica"] = 3.0
        };

        public double SepalLengthCm { get; private set; }
        public double SepalWidthCm { get; private set; }
        public double PetalLengthCm { get; private set; }
        public double PetalWidthCm { get; private set; }
        public double Species { get; private set; }

        public static IrisDataRow For(IReadOnlyList<string> row)
        {
            return new IrisDataRow
            {
                SepalLengthCm = double.Parse(row[1]),
                SepalWidthCm = double.Parse(row[2]),
                PetalLengthCm = double.Parse(row[3]),
                PetalWidthCm = double.Parse(row[4]),
                Species = SpeciesToDouble(row[5])
            };
        }

        private static double SpeciesToDouble(string species)
        {
            SpeciesCategoriesToDouble.TryGetValue(species, out var doubleRepresentation);
            return doubleRepresentation;
        }
    }
}