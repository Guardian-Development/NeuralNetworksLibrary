using System;
using Microsoft.Extensions.Logging;

namespace NeuralNetworks.Library.Logging
{
    public static class LoggerProvider
    {
        private static ILogger Log => Logger.CreateLogger(typeof(LoggerFactory));
        private static ILoggerFactory Logger { get; set; } = new LoggerFactory();

        internal static ILogger For<T>() => Logger.CreateLogger<T>();
        internal static ILogger For(Type type) => Logger.CreateLogger(type);

        public static ILoggerFactory InitialiseLoggingForNeuralNetworksLibrary(this ILoggerFactory factory)
        {
            Logger = factory;
            Log.LogDebug("Configured logging for the Neural Networks library");
            return Logger; 
        }
    }
}
