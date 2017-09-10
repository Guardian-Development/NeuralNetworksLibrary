using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace NeuralNetworks.Examples.FraudDetection.Web.Extensions
{
    public static class IServiceCollectionExtensions
    {
        public static TSettings BindApplicationSettings<TSettings>(
            this IServiceCollection services,
            IConfigurationSection configuration) where TSettings : class, new()
        {
            if (services == null) throw new ArgumentNullException(nameof(services));
            if (configuration == null) throw new ArgumentNullException(nameof(configuration));

            var settings = new TSettings();
            configuration.Bind(settings);
            services.AddSingleton(settings);
            return settings;
        }
    }
}