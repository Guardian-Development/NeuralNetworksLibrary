using System;
using System.Collections.Generic;

namespace NeuralNetworks.Library.Extensions
{
    public static class EnumerableExtensions
    {
        public static void ApplyInReverse<TEntity>(
            this List<TEntity> source, 
            Action<TEntity> action)
        {
            var sourceCount = source.Count; 
            for (var i = sourceCount - 1; i >= 0; i--)
            {
                action.Invoke(source[i]);
            }
        }

        public static void ForEach<TEntity>(
            this IEnumerable<TEntity> source, 
            Action<TEntity> action)
        {
            foreach(var entity in source)
            {
                action.Invoke(entity); 
            }
        }
    }
}