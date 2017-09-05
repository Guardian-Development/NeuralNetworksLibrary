using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NeuralNetworks.Examples.FraudDetection.Services;
using NeuralNetworks.Examples.FraudDetection.Services.Application;

namespace NeuralNetworks.Examples.FraudDetection.Web.Pages
{
    public class PerformTrainingModel : PageModel
    {
        private readonly NeuralNetworkService neuralNetworkService;

        public PerformTrainingModel(NeuralNetworkService neuralNetworkService)
        {
            this.neuralNetworkService = neuralNetworkService;
        }

        public void OnGet()
        {
        }

        public async void RequestTraining(int epochAmount)
        {
            await neuralNetworkService.BeginTraining(epochAmount);
        }
    }
}