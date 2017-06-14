using System.Collections.Generic;
using Xunit;

namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class ListAssertionHelpers
    {
        public static void AssertEqualLength<TSource, TCompare>(List<TSource> source, List<TCompare> compare)
        {
            Assert.True(source.Count == compare.Count, "Given lists are not equal in length");
        }
    }
}