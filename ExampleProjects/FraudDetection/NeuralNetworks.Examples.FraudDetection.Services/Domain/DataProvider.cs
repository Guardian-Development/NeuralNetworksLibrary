using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.Extensions.Options;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public class DataProvider 
    {
        public List<TrainingDataSet> TrainingData { get; }
            = new List<TrainingDataSet>();

        public List<BankTransaction> TestingData { get; }
            = new List<BankTransaction>(); 

        private readonly Lazy<IList<BankTransaction>> allDataRows; 

        public DataProvider(IOptions<DataSourceConfiguration> dataSource)
        {
            allDataRows = new Lazy<IList<BankTransaction>>(
                () => CsvFileReader.ReadFromFile(
                dataSource.Value.FileLocation, 
                BankTransaction.For));
        }
    }

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
                while(!EndOfFile(streamReader, out var currentRow))
                {
                    readRowsAsEntity.Add(ProcessRowOfFile(currentRow, rowToEntity)); 
                }
            }

            return readRowsAsEntity;
        }

        private static bool EndOfFile(StreamReader streamReader, out string currentRow)
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