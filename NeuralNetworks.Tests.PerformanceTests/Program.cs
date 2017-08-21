using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using NeuralNetworks.Library;
using NeuralNetworks.Library.Components.Activation;
using NeuralNetworks.Library.Extensions;

namespace NeuralNetworks.Tests.PerformanceTests
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var switcher = new BenchmarkSwitcher(new[] {
                typeof(FeedForwardPerformanceComparisonContainer),
                typeof(BackPropagationPerformanceComparisonContainer)
            });
            switcher.Run(args);
        }
    }
}
