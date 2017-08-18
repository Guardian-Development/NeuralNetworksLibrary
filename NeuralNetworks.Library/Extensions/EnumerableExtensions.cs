using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

        public static void ForEach<TEntity>(
            this IList<TEntity> source,
            Action<TEntity, int> action)
        {
            for(var i = 0; i < source.Count(); i++)
            {
                var entity = source[i];
                action.Invoke(entity, i); 
            }
        }

        public static void ParallelForEach<TEntity>(
            this IList<TEntity> source, 
            Action<TEntity, int> action,
            ParallelOptions parallelOptions)
        {
            Parallel.For(
                fromInclusive: 0, 
                toExclusive: source.Count, 
                body: (index, loopState) => 
                {
                    var entity = source[index]; 
                    action.Invoke(entity, index); 
                });
        }
    }
}