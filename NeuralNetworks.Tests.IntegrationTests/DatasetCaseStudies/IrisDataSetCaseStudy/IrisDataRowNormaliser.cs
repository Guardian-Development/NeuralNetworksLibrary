using System.Linq;

namespace NeuralNetworks.Tests.IntegrationTests.DatasetCaseStudies.IrisDatasetCaseStudy
{
    public static class IrisDataRowNormaliser
    {
        private static double MinSepalLengthCm => 
            IrisDataSet.AllRows
                .Select(row => row.SepalLengthCm)
                .Min(); 

        private static double MaxSepalLengthCm => 
            IrisDataSet.AllRows
                .Select(row => row.SepalLengthCm)
                .Max(); 
        
        private static double MinSepalWidthCm => 
            IrisDataSet.AllRows
                .Select(row => row.SepalWidthCm)
                .Min();
        
        private static double MaxSepalWidthCm => 
            IrisDataSet.AllRows
                .Select(row => row.SepalWidthCm)
                .Max(); 

         private static double MinPetalLengthCm => 
            IrisDataSet.AllRows
                .Select(row => row.PetalLengthCm)
                .Min(); 

        private static double MaxPetalLengthCm => 
            IrisDataSet.AllRows
                .Select(row => row.PetalLengthCm)
                .Max(); 

        private static double MinPetalWidthCm => 
            IrisDataSet.AllRows
                .Select(row => row.PetalWidthCm)
                .Min(); 

        private static double MaxPetalWidthCm => 
            IrisDataSet.AllRows
                .Select(row => row.PetalWidthCm)
                .Max(); 


        public static IrisDataRow NormaliseDataRow(IrisDataRow row)
        {
            return IrisDataRow.For(
                row.Id, 
                row.SepalLengthCm.NormaliseValue(MinSepalLengthCm, MaxSepalLengthCm),
                row.SepalWidthCm.NormaliseValue(MinSepalWidthCm, MaxSepalWidthCm),
                row.PetalLengthCm.NormaliseValue(MinPetalLengthCm, MaxPetalLengthCm),
                row.PetalWidthCm.NormaliseValue(MinPetalWidthCm, MaxPetalWidthCm),
                row.Species); 
        }

        private static double NormaliseValue(this double value, double min, double max)
        {
            return (value - min) / (max - min); 
        }
    }
}