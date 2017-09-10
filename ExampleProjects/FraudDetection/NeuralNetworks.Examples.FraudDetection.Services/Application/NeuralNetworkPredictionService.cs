using System;
using System.Linq;
using System.Threading.Tasks;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library.Extensions;

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
            return dataProvider.TrainingData
                .Aggregate(new NeuralNetworkPredictionsReport(0, 0), MakePredictionRecordingResult);
        }

        private NeuralNetworkPredictionsReport MakePredictionRecordingResult(
            NeuralNetworkPredictionsReport report,
            BankTransaction transaction)
        {
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