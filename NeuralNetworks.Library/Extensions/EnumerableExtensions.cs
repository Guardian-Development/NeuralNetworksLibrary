using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetworks.Library.Extensions
{
    public static class EnumerableExtensions
    {
        public static void ApplyInReverse<TEntity>(
            this IList<TEntity> source, 
            Action<TEntity> action)
        {
            var sourceCount = source.Count; 
            for (var i = sourceCount - 1; i >= 0; i--)
            {
                action.Invoke(source[i]);
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
                parallelOptions: parallelOptions, 
                body: (index, loopState) => 
                {
                    var entity = source[index]; 
                    action.Invoke(entity, index); 
                });
        }

        public static void ParallelForEach<TEntity>(
            this IEnumerable<TEntity> source,
            Action<TEntity> action, 
            ParallelOptions parallelOptions)
        {
            Parallel.ForEach<TEntity>(
                source, 
                parallelOptions,
                action);
        }

        public static IEnumerable<TResult> ParallelZip<TSourceEntity, TReferencedEntity, TResult>(
            this IList<TSourceEntity> source, 
            IList<TReferencedEntity> target, 
            Func<TSourceEntity, TReferencedEntity, TResult> zipFunc,
            ParallelOptions parallelOptions)
        {
            var zippedList = new List<TResult>(); 
            var lockObject = new object(); 

            Parallel.For(
                fromInclusive: 0,
                toExclusive: source.Count, 
                parallelOptions: parallelOptions, 
                body: (index, loopState) => {
                    var zipResult = zipFunc(source[index], target[index]);
                    lock(lockObject) {
                        zippedList.Add(zipResult); 
                    }
                }
            ); 

            return zippedList;
        }

        public static double ParallelSum(
            this IEnumerable<double> source, 
            ParallelOptions parallelOptions)
        {
            var sum = 0.0; 
            var lockObject = new object(); 

            Parallel.ForEach(
                source, 
                () => 0.0, 
                (input, loopState, partialResult) => input + partialResult, 
                localPartialSum => {
                    lock (lockObject) sum += localPartialSum; 
                }
            ); 

            return sum; 
        }
    }
}