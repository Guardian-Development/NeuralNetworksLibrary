using System;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public sealed class NeuralNetworkPredictionsReport
    {
        public int TotalCorrectPredictions { get; }
        public int TotalIncorrectPredictions { get; }
        public int IncorrectFraudulentPredictions { get; }
        public int IncorrectLegitimatePredictions { get; }

        private NeuralNetworkPredictionsReport(
            int numberOfCorrectPredictions, 
            int numberOfIncorrectPredictions,
            int incorrectFraudulentPredictions,
            int incorrectLegitimatePredictions)
        {
            TotalCorrectPredictions = numberOfCorrectPredictions; 
            TotalIncorrectPredictions = numberOfIncorrectPredictions; 
            IncorrectFraudulentPredictions = incorrectFraudulentPredictions; 
            IncorrectLegitimatePredictions = incorrectLegitimatePredictions; 
        }

        public override string ToString()
        {
            return $"Correct: {TotalCorrectPredictions}. Incorrect: {TotalIncorrectPredictions}";
        }

        internal static NeuralNetworkPredictionsReport EmptyReport() 
            => new NeuralNetworkPredictionsReport(0, 0, 0, 0); 

        internal static NeuralNetworkPredictionsReport UpdateFor(
            NeuralNetworkPredictionsReport previousReport, 
            BankTransactionClass predictionClass, 
            bool predictionWasCorrect)
        {
            if(predictionWasCorrect)
            {
                return new NeuralNetworkPredictionsReport(
                    previousReport.TotalCorrectPredictions + 1,
                    previousReport.TotalIncorrectPredictions, 
                    previousReport.IncorrectFraudulentPredictions,
                    previousReport.IncorrectLegitimatePredictions); 
            }

            if(predictionClass == BankTransactionClass.Fraudulent)
            {
                return new NeuralNetworkPredictionsReport(
                    previousReport.TotalCorrectPredictions,
                    previousReport.TotalIncorrectPredictions + 1,
                    previousReport.IncorrectFraudulentPredictions + 1,
                    previousReport.IncorrectLegitimatePredictions); 
            }

            if(predictionClass == BankTransactionClass.Legitimate)
            {
                return new NeuralNetworkPredictionsReport(
                    previousReport.TotalCorrectPredictions,
                    previousReport.TotalIncorrectPredictions + 1,
                    previousReport.IncorrectFraudulentPredictions,
                    previousReport.IncorrectLegitimatePredictions + 1);
            }

            throw new InvalidOperationException($"Transaction class {predictionClass} is not supported."); 
        }
    }
}