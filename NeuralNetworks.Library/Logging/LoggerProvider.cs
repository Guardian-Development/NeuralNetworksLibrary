﻿using System;
using Microsoft.Extensions.Logging;

namespace NeuralNetworks.Library.Logging
{
    public static class LoggerProvider
    {
        private static ILogger Log => Logger.CreateLogger(typeof(LoggerFactory));
        private static ILoggerFactory Logger { get; set; } = new LoggerFactory();

        internal static ILogger For<T>() => Logger.CreateLogger<T>();
        internal static ILogger For(Type type) => Logger.CreateLogger(type);

        public static void InitialiseLoggingForNeuralNetworksLibrary(this ILoggerFactory factory)
        {
            Logger = factory;
            Log.LogInformation("Configured logging for the Neural Networks library");
        }
    }
}