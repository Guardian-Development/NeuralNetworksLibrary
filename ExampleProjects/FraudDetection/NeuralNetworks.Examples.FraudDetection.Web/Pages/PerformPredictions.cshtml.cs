using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NeuralNetworks.Examples.FraudDetection.Services;
using NeuralNetworks.Examples.FraudDetection.Services.Application;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Examples.FraudDetection.Web.Models;

namespace NeuralNetworks.Examples.FraudDetection.Web.Pages
{
    public class PerformPredictionsModel : PageModel
    {
        private readonly NeuralNetworkPredictionService networkPredictionService; 

        public PerformPredictionsModel(NeuralNetworkPredictionService networkPredictionService)
        {
            this.networkPredictionService = networkPredictionService; 
        }

        public PredictionsReport PredictionsReport { get; set; }
        
        public void OnGet(PredictionsReport predictionsReport)
        {
            PredictionsReport = predictionsReport; 
        }

        public IActionResult OnPostPredict()
        {
             var predictionsReport = networkPredictionService.RunPredictions();
             return RedirectToPage(new PredictionsReport(){
                TotalCorrectPredictions = predictionsReport.TotalCorrectPredictions,
                TotalIncorrectPredictions = predictionsReport.TotalIncorrectPredictions,
                IncorrectFraudulentPredictions = predictionsReport.IncorrectFraudulentPredictions,
                IncorrectLegitimatePredictions = predictionsReport.IncorrectLegitimatePredictions});
        }
    }
}