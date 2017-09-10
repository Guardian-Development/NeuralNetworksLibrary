using System;
using System.Threading.Tasks;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;

namespace NeuralNetworks.Examples.FraudDetection.Services.Application
{
    public sealed class NeuralNetworkPredictionService
    {
        private readonly NeuralNetworkAccessor networkAccessor;
        private readonly DataProvider dataProvider;

        public NeuralNetworkPredictionService(
            NeuralNetworkAccessor networkAccessor,
            DataProvider dataProvider)
        {
            this.networkAccessor = networkAccessor; 
            this.dataProvider = dataProvider;
        }

        public NeuralNetworkPredictionsReport RunPredictions()
        {
            throw new NotImplementedException(); 
        }
    }
}