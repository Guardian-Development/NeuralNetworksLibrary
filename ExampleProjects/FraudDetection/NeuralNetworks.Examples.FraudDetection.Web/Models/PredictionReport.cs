using System;

namespace NeuralNetworks.Examples.FraudDetection.Web.Models
{
    public class PredictionReport 
    {
        public int NumberOfCorrectPredictions { get; set; }
        public int NumberOfIncorrectPredictions { get; set; }

        public string CorrectPercentage 
        {
            get {
                var total = NumberOfCorrectPredictions + NumberOfIncorrectPredictions;
                if(total == 0) return $"{0:00.00}%";  

                var percentage = (Convert.ToDouble(NumberOfCorrectPredictions) / Convert.ToDouble(total)) * 100; 
                return $"{((percentage)):00.00}%";
            }
        }
           
    }
}