using System.Collections.Generic;
using Microsoft.Extensions.Options;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public class DataProvider 
    {
        public List<TrainingDataSet> TrainingData { get; }
            = new List<TrainingDataSet>(); 

        private readonly DataSourceConfiguration dataSource;

        public DataProvider(IOptions<DataSourceConfiguration> dataSource)
        {
            this.dataSource = dataSource.Value;
        }
    }
}