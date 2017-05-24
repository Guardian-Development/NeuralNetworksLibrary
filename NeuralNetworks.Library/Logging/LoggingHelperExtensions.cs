namespace NeuralNetworks.Library.Logging
{
    public static class LoggingHelperExtensions
    {
        internal static string LogArray<TEntity>(this TEntity[] entityArray)
        {
            return "[" + string.Join(",", entityArray) + "]";
        }
    }
}
