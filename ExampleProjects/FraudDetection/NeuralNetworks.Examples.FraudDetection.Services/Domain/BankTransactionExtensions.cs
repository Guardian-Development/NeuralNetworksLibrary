using System;
using System.Linq;
using NeuralNetworks.Library.Data;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    internal static class BankTransactionExtensions
    {
        public static TrainingDataSet ToTrainingData(this BankTransaction bankTransaction)
        {
            var inputs = new [] {
                bankTransaction.Amount, 
                bankTransaction.DataPoint1,
                bankTransaction.DataPoint2,
                bankTransaction.DataPoint3,
                bankTransaction.DataPoint4,
                bankTransaction.DataPoint5,
                bankTransaction.DataPoint6,
                bankTransaction.DataPoint7,
                bankTransaction.DataPoint8,
                bankTransaction.DataPoint9,
                bankTransaction.DataPoint10,
                bankTransaction.DataPoint11,
                bankTransaction.DataPoint12,
                bankTransaction.DataPoint13,
                bankTransaction.DataPoint14,
                bankTransaction.DataPoint15,
                bankTransaction.DataPoint16,
                bankTransaction.DataPoint17,
                bankTransaction.DataPoint18,
                bankTransaction.DataPoint19,
                bankTransaction.DataPoint20,
                bankTransaction.DataPoint21,
                bankTransaction.DataPoint22,
                bankTransaction.DataPoint23,
                bankTransaction.DataPoint24,
                bankTransaction.DataPoint25,
                bankTransaction.DataPoint26,
                bankTransaction.DataPoint27,
                bankTransaction.DataPoint28
            }; 

            var outputs = Enumerable.Repeat(0.0, 2).ToList(); 
            outputs[Convert.ToInt32(bankTransaction.Class)] = 1;

            return TrainingDataSet.For(inputs.ToArray(), outputs.ToArray());
        }
    }
}
