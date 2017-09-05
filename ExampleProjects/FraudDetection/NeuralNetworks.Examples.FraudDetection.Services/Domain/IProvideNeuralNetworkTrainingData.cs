using System.Collections.Generic;
using System;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public interface IProvideNeuralNetworkTrainingData
    {
        IList<TrainingDataSet> TrainingData { get; }
    }

    internal class NeuralNetworkTrainingDataFromCsvFile : IProvideNeuralNetworkTrainingData
    {
        private readonly DataSetConfiguration dataSetConfiguration; 

        public NeuralNetworkTrainingDataFromCsvFile(DataSetConfiguration dataSetConfiguration)
        {
            this.dataSetConfiguration = dataSetConfiguration;
        }
        
        public IList<TrainingDataSet> TrainingData => throw new NotImplementedException();
    }
}