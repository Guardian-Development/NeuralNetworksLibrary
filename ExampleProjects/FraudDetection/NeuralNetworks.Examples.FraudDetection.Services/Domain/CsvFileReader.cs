using System;
using System.Collections.Generic;
using System.IO;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
      public static class CsvFileReader 
    {
        public static IList<TEntity> ReadFromFile<TEntity>(
            string csvFilepath, 
            Func<string[], TEntity> rowToEntity,
            bool containsHeaders = true)
        {
            var readRowsAsEntity = new List<TEntity>(); 

            using(var streamReader = File.OpenText(csvFilepath))
            {
                if(containsHeaders)
                {
                    ReadRowOfFileIfNotAtEnd(streamReader, out _);
                }

                while(!ReadRowOfFileIfNotAtEnd(streamReader, out var currentRow))
                {
                    readRowsAsEntity.Add(ProcessRowOfFile(currentRow, rowToEntity)); 
                }
            }

            return readRowsAsEntity;
        }

        private static bool ReadRowOfFileIfNotAtEnd(StreamReader streamReader, out string currentRow)
        {
            currentRow = streamReader.ReadLine(); 
            return currentRow == null ? true : false; 
        }

        private static TEntity ProcessRowOfFile<TEntity>(
            string line, 
            Func<string[], TEntity> rowToEntity)
        {
            var dataColumns = line.Split(',');
            return rowToEntity.Invoke(dataColumns); 
        }
    }
}