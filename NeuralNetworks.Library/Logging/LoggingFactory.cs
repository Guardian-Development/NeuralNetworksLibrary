using System;
using Microsoft.Extensions.Logging;

namespace NeuralNetworks.Library.Logging
{
    public static class ApplicationLogging
    {
        private static ILogger Log => Logger.CreateLogger(typeof(ApplicationLogging));
        private static ILoggerFactory Logger { get; set; } = new LoggerFactory();

        public static ILogger For<T>() => Logger.CreateLogger<T>();
        public static ILogger For(Type type) => Logger.CreateLogger(type);

        public static void InitialiseLoggingForNeuralNetworksLibrary(this ILoggerFactory factory)
        {
            Logger = factory;
            Log.LogInformation("Configured logging for the Neural Networks library");
        }
    }
}
