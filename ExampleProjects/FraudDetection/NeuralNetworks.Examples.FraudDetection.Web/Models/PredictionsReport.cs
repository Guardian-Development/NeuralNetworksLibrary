using System;

namespace NeuralNetworks.Examples.FraudDetection.Web.Models
{
    public class PredictionsReport 
    {
        public int TotalCorrectPredictions { get; set; }
        public int TotalIncorrectPredictions { get; set; }

        public string PercentageOfPredictionsCorrect 
        {
            get {
                var total = TotalCorrectPredictions + TotalIncorrectPredictions;
                if(total == 0) return $"{0:00.00}%";  

                var percentage = (Convert.ToDouble(TotalCorrectPredictions) / Convert.ToDouble(total)) * 100; 
                return $"{((percentage)):00.00}%";
            }
        }

        public int IncorrectFraudulentPredictions { get; set; }
        public int IncorrectLegitimatePredictions { get; set; }

        public string PercentageOfIncorrectPredictionsFraudulent 
        {
            get 
            {
                var total = IncorrectFraudulentPredictions + IncorrectLegitimatePredictions;
                if(total == 0) return $"{0:00.00}%";  

                var percentage = (Convert.ToDouble(IncorrectFraudulentPredictions) / Convert.ToDouble(total)) * 100; 
                return $"{((percentage)):00.00}%";
            }
        }
           
    }
}