using Microsoft.Extensions.DependencyInjection;
using NeuralNetworks.Examples.FraudDetection.Services.Domain;
using NeuralNetworks.Library.Training;

namespace NeuralNetworks.Examples.FraudDetection.Services
{
    public static class ServiceLayerConfiguration
    {
        public static IServiceCollection ConfigureServiceLayer(
            this IServiceCollection services, 
            DataSetConfiguration configuration)
        {
            services.AddSingleton(configuration); 
            services.AddSingleton<NeuralNetworkConfiguration>(); 
            services.AddSingleton<IProvideNeuralNetworkTrainingData, NeuralNetworkTrainingDataFromCsvFile>();
            
            return services; 
        }
    }
}