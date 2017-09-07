using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NeuralNetworks.Examples.FraudDetection.Services;
using NeuralNetworks.Examples.FraudDetection.Services.Application;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;

namespace NeuralNetworks.Examples.FraudDetection.Web.Pages
{
    public class PerformTrainingModel : PageModel
    {
        private readonly NeuralNetworkTrainingService networkTrainingService;
        private readonly NeuralNetworkTrainingConfiguration trainingConfiguration;

        public PerformTrainingModel(
            NeuralNetworkTrainingService networkTrainingService,
            NeuralNetworkTrainingConfiguration trainingConfiguration)
        {
            this.networkTrainingService = networkTrainingService;
            this.trainingConfiguration = trainingConfiguration;
        }

        public void OnGet()
        {
        }

        public async void RequestTraining(int epochAmount)
        {
            await networkTrainingService.TrainConfiguredNetworkForEpochs(epochAmount, trainingConfiguration); 
        }
    }
}