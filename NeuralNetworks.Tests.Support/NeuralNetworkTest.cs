using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Tests.Support
{
    public class NeuralNetworkTest
    {
        private LoggerFactory Logger { get; } = new LoggerFactory();

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