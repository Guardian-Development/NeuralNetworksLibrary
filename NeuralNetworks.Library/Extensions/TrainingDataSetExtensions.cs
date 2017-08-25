using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Library.Extensions
{
    public static class TrainingDataSetExtensions
    {
        public static IEnumerable<TrainingDataSet> BuildTrainingDataForAllInputs(
            IEnumerable<double[]> inputs, 
            IEnumerable<double[]> outputs)
        {
            return inputs.Zip(outputs, (input, output) => TrainingDataSet.For(input, output)); 
        }
    }
}