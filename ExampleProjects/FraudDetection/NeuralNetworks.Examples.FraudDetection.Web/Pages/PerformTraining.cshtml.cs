using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NeuralNetworks.Examples.FraudDetection.Services; 

namespace NeuralNetworks.Examples.FraudDetection.Web.Pages
{
    public class PerformTrainingModel : PageModel
    {
        private readonly DataSetConfiguration dataSetConfiguration;

        public PerformTrainingModel(DataSetConfiguration dataSetConfiguration)
        {
            this.dataSetConfiguration = dataSetConfiguration;
        }

        public void OnGet()
        {
        }
    }
}