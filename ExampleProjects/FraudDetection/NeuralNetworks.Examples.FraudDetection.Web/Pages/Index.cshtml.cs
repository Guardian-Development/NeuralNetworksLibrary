﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using NeuralNetworks.Examples.FraudDetection.Services; 

namespace NeuralNetworks.Examples.FraudDetection.Web.Pages
{
    public class IndexModel : PageModel
    {
        public string TestFullStack { get; private set; }

        public void OnGet()
        {
            TestFullStack = Class1.TestFSharp(); 
        }
    }
}