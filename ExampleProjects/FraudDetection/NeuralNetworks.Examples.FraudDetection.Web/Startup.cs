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

namespace NeuralNetworks.Examples.FraudDetection.Web
{
    public class Startup
    {
        private const string FraudDataSetConfigurationSection = "DataSetConfiguration"; 
        
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddMvc();

             var dataSetConfiguration = services.BindApplicationSettings<DataSetConfiguration>(
                Configuration.GetSection(FraudDataSetConfigurationSection)); 

            services.ConfigureServiceLayer(dataSetConfiguration);
            services.AddSingleton(typeof(NeuralNetworkService)); 
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(IApplicationBuilder app, IHostingEnvironment env)
        {
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
