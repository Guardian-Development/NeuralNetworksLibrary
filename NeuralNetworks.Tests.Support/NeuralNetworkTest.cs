using System;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;
using NeuralNetworks.Library.NetworkInitialisation;

namespace NeuralNetworks.Tests.Support
{
    public class NeuralNetworkTest
    {
        private LoggerFactory Logger { get; } = new LoggerFactory();

        protected ILogger LogFor(Type type) => Logger.CreateLogger(type); 
        
        protected IProvideRandomNumberGeneration PredictableGenerator => 
            PredictableRandomNumberGenerator.Create(); 

        protected NeuralNetworkTest()
        {
            ConfigureLoggingForTest(); 
        }

        private void ConfigureLoggingForTest()
        {
            var testName = GetType().Name; 
            Logger
                .AddConsole(LogLevel.Debug)
                .AddFile($"Logs/{testName}-{{Date}}.txt", LogLevel.Trace)
                .InitialiseLoggingForNeuralNetworksLibrary();
        }
    }
}