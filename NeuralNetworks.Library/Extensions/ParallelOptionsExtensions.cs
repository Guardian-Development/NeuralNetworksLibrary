using System.Threading.Tasks;

namespace NeuralNetworks.Library.Extensions
{
    public static class ParallelOptionsExtensions
    {
        public static ParallelOptions SingleThreadedOptions => 
            new ParallelOptions {
                MaxDegreeOfParallelism = 1
            };

        public static ParallelOptions UnrestrictedMultiThreadedOptions => 
            new ParallelOptions(); 

        
        public static ParallelOptions MultiThreadedOptions(int maxNumberOfThreads) => 
            new ParallelOptions {
                MaxDegreeOfParallelism = maxNumberOfThreads
            }; 
    }
}