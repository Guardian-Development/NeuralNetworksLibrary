using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.IrisDatasetCaseStudy
{
    public sealed class IrisDataRow 
    {
        private IrisDataRow(
            int id,
            double sepalLengthCm, 
            double sepalWithCm,
            double petalLengthCm,
            double petalWithCm,
            double species)
        {
            Id = id; 
            SepalLengthCm = sepalLengthCm; 
            SepalWidthCm = sepalWithCm; 
            PetalLengthCm = petalLengthCm; 
            PetalWidthCm = petalWithCm; 
            Species = species; 
        }

        public int Id { get; }
        public double SepalLengthCm { get; }
        public double SepalWidthCm { get; }
        public double PetalLengthCm { get; }
        public double PetalWidthCm { get; }
        public double Species { get; }

        public double[] PredictionDataPoints => 
            new [] { SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm }; 

        public override bool Equals(object obj) 
        {
            var objAsIrisDataRow = obj as IrisDataRow; 
            if(objAsIrisDataRow == null)
            {
                return false; 
            }

            return Equals(objAsIrisDataRow); 
        }

        public bool Equals(IrisDataRow row)
        {
            return this.Id == row.Id; 
        }

        public override int GetHashCode()
        {
            return Id; 
        }

        public static TrainingDataSet TrainingDataFromRow(IrisDataRow row)
        {
            var species = Enumerable.Repeat(0.0, 3).ToArray();
            species[Convert.ToInt32(row.Species) - 1] = 1.0;

            return TrainingDataSet.For(
                        row.PredictionDataPoints, 
                        species.ToArray());
        }

        public static IrisDataRow For(
            int id,
            double sepalLengthCm, 
            double sepalWithCm,
            double petalLengthCm,
            double petalWithCm,
            double species) 
        {
            return new IrisDataRow(id, sepalLengthCm, sepalWithCm, petalLengthCm, petalWithCm, species); 
        }
    }
}