using System;
using System.Collections.Generic;
using System.Linq; 

namespace NeuralNetworks.Tests.Support.Assertors
{
    public class OrderedListAssertor<T> : IAssert<IEnumerable<T>>
    {
        public IList<IAssert<T>> Assertors { get; } = new List<IAssert<T>>();

        public OrderedListAssertor()
        {}

        public void Assert(IEnumerable<T> actualItem)
        {
            var actualItemList= actualItem.ToList(); 

            for(int i = 0; i < actualItemList.Count; i++)
            {
                var assertor = Assertors[i]; 
                var item = actualItemList[i]; 
                assertor.Assert(item);
            }
            
            Xunit.Assert.True(
                Assertors.Count == actualItem.Count(), 
                $"Expected: {Assertors.Count} Recevied: {actualItem.Count()}");
        }
    }
}
