using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using NeuralNetworks.Examples.FraudDetection.Web.Extensions;
using NeuralNetworks.Examples.FraudDetection.Services.Application;
using NeuralNetworks.Examples.FraudDetection.Services;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;
using Microsoft.Extensions.Logging;

namespace NeuralNetworks.Examples.FraudDetection.Web
{
    public class Startup
    {
        private const string DataSourceConfigurationSection = "DataSourceConfiguration"; 
        private const string NeuralNetworkTrainingConfigurationSection = "NeuralNetworkTrainingConfiguration"; 
        
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddMvc();

            services.Configure<DataSourceConfiguration>(Configuration.GetSection(DataSourceConfigurationSection)) 
                    .Configure<NeuralNetworkTrainingConfiguration>(Configuration.GetSection(NeuralNetworkTrainingConfigurationSection))
                    .ConfigureServiceLayer()
                    .AddTransient(typeof(NeuralNetworkTrainingService))
                    .AddTransient(typeof(NeuralNetworkPredictionService));
        }

        public void Configure(IApplicationBuilder app, IHostingEnvironment env, ILoggerFactory loggerFactory)
        {
            loggerFactory.ConfigureServiceLayerLogging(); 
            
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
            }

            app.UseStaticFiles();

            app.UseMvc(routes =>
            {
                routes.MapRoute(
                    name: "default",
                    template: "{controller}/{action=Index}/{id?}");
            });
        }
    }
}
