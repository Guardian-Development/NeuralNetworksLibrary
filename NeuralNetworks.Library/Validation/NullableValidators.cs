using System;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Library.Validation
{
    internal static class NullableValidators
    {
        internal static void ValidateNotNull<TEntity>(TEntity entity, [CallerMemberName] string callerName = null)
        {
            if (entity == null)
            {
                throw new InvalidOperationException(
                    $"{typeof(TEntity).Name} must be set before calling {callerName}");
            }
        }
    }
}
