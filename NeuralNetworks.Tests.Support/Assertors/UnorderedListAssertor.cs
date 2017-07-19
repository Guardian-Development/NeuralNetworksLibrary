using System;
using System.Collections.Generic;
using System.Linq; 

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class UnorderedListAssertor<TKey, T> : IAssert<IEnumerable<T>>
    {
        public IDictionary<TKey, IAssert<T>> Assertors { get; }
            = new Dictionary<TKey, IAssert<T>>();

        private readonly Func<T, TKey> getKeyForAssertor;

        public UnorderedListAssertor(Func<T, TKey> getKeyForAssertor)
        {
            this.getKeyForAssertor = getKeyForAssertor;
        }

        public void Assert(IEnumerable<T> actualItem)
        {
            actualItem = actualItem.ToList(); 
            foreach(var item in actualItem)
            {
                var assertor = AssertorFor(item); 
                assertor.Assert(item);
            }

            Xunit.Assert.True(
                Assertors.Count == actualItem.Count(), 
                $"Expected: {Assertors.Count} Recevied: {actualItem.Count()}");
        }

        private IAssert<T> AssertorFor(T itemToAssert)
        {
            var keyForAssertor = getKeyForAssertor(itemToAssert); 
            if(Assertors.TryGetValue(keyForAssertor, out var assertor))
            {
                return assertor; 
            }

            throw new InvalidOperationException(
                $"No assertor registered under key : {keyForAssertor} for value: {itemToAssert}"); 
        }
    }
}
