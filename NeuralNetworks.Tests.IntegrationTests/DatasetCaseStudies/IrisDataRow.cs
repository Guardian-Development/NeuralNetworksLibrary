using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies
{
    public class IrisDataRow
    {
        public double SepalLengthCm { get; private set; }
        public double SepalWidthCm { get; private set; }
        public double PetalLengthCm { get; private set; }
        public double PetalWidthCm { get; private set; }
        public string Species { get; private set; }

        public static IrisDataRow For(IReadOnlyList<string> row)
        {
            return new IrisDataRow
            {
                SepalLengthCm = double.Parse(row[1]),
                SepalWidthCm = double.Parse(row[2]),
                PetalLengthCm = double.Parse(row[3]),
                PetalWidthCm = double.Parse(row[4]),
                Species = row[5]
            };
        }
    }
}