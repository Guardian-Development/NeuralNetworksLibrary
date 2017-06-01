using System;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Tests.Support
{
    public class NeuralNetworkTest
    {
        private LoggerFactory Logger { get; } = new LoggerFactory();
        protected ILogger LogFor(Type type) => Logger.CreateLogger(type); 

        protected NeuralNetworkTest()
        {
            ConfigureLoggingForTest(); 
        }

        private void ConfigureLoggingForTest()
        {
            Logger
                .AddConsole(LogLevel.Information)
                .InitialiseLoggingForNeuralNetworksLibrary();
        }
    }
}