using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;

namespace NeuralNetworks.Tests.Support
{
    public static class ReadCsv
    {
        public static IEnumerable<TEntity> FromEmbeddedResource<TEntity>(
            Assembly assembley,
            string resourceName,
            Func<string[], TEntity> converter,
            bool headerIncluded = true)
        {
            using (var stream = assembley.GetManifestResourceStream(resourceName))
            using (var reader = new StreamReader(stream))
            {
                var entities = new List<TEntity>();

                if (headerIncluded) reader.ReadLine();

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine().Split(',');
                    entities.Add(converter.Invoke(line));
                }

                return entities; 
            }
        }
    }
}