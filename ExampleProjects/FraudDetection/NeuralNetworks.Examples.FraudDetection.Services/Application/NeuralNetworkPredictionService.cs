using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Examples.FraudDetection.Services.Application
{
    public sealed class NeuralNetworkPredictionService
    {
        private readonly ILogger<NeuralNetworkPredictionService> logger; 
        private readonly NeuralNetworkAccessor networkAccessor;
        private readonly DataProvider dataProvider;

        public NeuralNetworkPredictionService(
            ILogger<NeuralNetworkPredictionService> logger,
            NeuralNetworkAccessor networkAccessor,
            DataProvider dataProvider)
        {
            this.logger = logger; 
            this.networkAccessor = networkAccessor; 
            this.dataProvider = dataProvider;
        }

        public NeuralNetworkPredictionsReport RunPredictions()
        {
            return dataProvider.TestingData
                .Aggregate(new NeuralNetworkPredictionsReport(0, 0), MakePredictionRecordingResult);
        }

        private NeuralNetworkPredictionsReport MakePredictionRecordingResult(
            NeuralNetworkPredictionsReport report,
            BankTransaction transaction)
        {
            logger.LogInformation($"Predicting new transaction. Current report: {report}"); 

            var networkPrediction = networkAccessor.TargetNetwork
                .PredictionFor(transaction.ToNetworkInputData(), 
                ParallelOptionsExtensions.UnrestrictedMultiThreadedOptions); 
            
            if(IsCorrectPrediction(networkPrediction, transaction.Class))
            {
                return new NeuralNetworkPredictionsReport(
                    report.NumberOfCorrectPredictions + 1, 
                    report.NumberOfIncorrectPredictions); 
            }

            return new NeuralNetworkPredictionsReport(
                report.NumberOfCorrectPredictions,
                report.NumberOfIncorrectPredictions + 1); 
        }

        private bool IsCorrectPrediction(double[] prediction, BankTransactionClass actualClass)
        {
            var predictedClass = Array.IndexOf(prediction, prediction.Max()) + 1;
            
            return (BankTransactionClass)predictedClass == actualClass ? true : false; 
        }
    }
}