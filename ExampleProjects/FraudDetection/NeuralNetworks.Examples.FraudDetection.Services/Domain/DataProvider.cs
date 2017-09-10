using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Options;
using NeuralNetworks.Examples.FraudDetection.Services.Configuration;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public class DataProvider 
    {
        public List<BankTransaction> TrainingData => trainingData.Value; 
        
        private readonly Lazy<List<BankTransaction>> trainingData; 

        public List<BankTransaction> TestingData => testingData.Value; 

        private readonly Lazy<List<BankTransaction>> testingData; 

        private readonly Lazy<IList<BankTransaction>> allDataRows; 

        private List<BankTransaction> TrainingDataRowsForClass(BankTransactionClass targetClass)
        {
            var dataRowsInClass = allDataRows.Value
                .Where(transaction => transaction.Class == targetClass)
                .ToList();
            
            return RandomSubsetOfDataSet(dataRowsInClass, rowCountOfSmallestTransactionCategory.Value / 2);
        }

        private List<TEntity> RandomSubsetOfDataSet<TEntity>(List<TEntity> dataSet, int subsetAmount)
        {
            var randomNumberGenerator = new Random(1); 

            return Enumerable
                        .Range(0, subsetAmount)
                        .Select(_ => dataSet[randomNumberGenerator.Next(0, dataSet.Count)])
                        .ToList();
        }

        private readonly Lazy<double> bankTransactionAmountMin;
        private readonly Lazy<double> bankTransactionAmountMax; 
        private readonly Lazy<int> rowCountOfSmallestTransactionCategory; 

        public DataProvider(IOptions<DataSourceConfiguration> dataSource)
        {
            allDataRows = new Lazy<IList<BankTransaction>>(
                () => CsvFileReader.ReadFromFile(
                dataSource.Value.FileLocation, 
                BankTransaction.For));
            
            bankTransactionAmountMin = new Lazy<double>(
                () => allDataRows.Value
                        .Select(r => r.Amount)
                        .Min());
            
            bankTransactionAmountMax = new Lazy<double>(
                () => allDataRows.Value
                        .Select(r => r.Amount)
                        .Max());
            
            rowCountOfSmallestTransactionCategory = new Lazy<int>(
                () => allDataRows.Value
                        .GroupBy(transaction => transaction.Class)
                        .Select(g => g.Count())
                        .Min());
            
            trainingData = new Lazy<List<BankTransaction>>(
               () => TrainingDataRowsForClass(BankTransactionClass.Fraudulent)
                .Concat(TrainingDataRowsForClass(BankTransactionClass.Legitimate))
                .Select(transaction => BankTransactionNormaliser
                    .NormaliseTransactionAmount(
                        transaction, 
                        bankTransactionAmountMin.Value, 
                        bankTransactionAmountMax.Value))
                .ToList()
            );

            testingData = new Lazy<List<BankTransaction>>(
                () => allDataRows.Value
                .Select(transaction => BankTransactionNormaliser
                    .NormaliseTransactionAmount(
                        transaction, 
                        bankTransactionAmountMin.Value, 
                        bankTransactionAmountMax.Value))
                .Except(TrainingData)
                .ToList()
            ); 
        }
    }
}