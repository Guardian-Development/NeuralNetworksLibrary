using System;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public enum BankTransactionCategory
    {
        Legitimate,
        Fraudulent
    }

    public class BankTransaction
    {
        public BankTransactionCategory Class { get; set; }

        public int TimeOfTransaction { get; set; }
        public double Amount { get; set; }

        public double DataPoint1 { get; set; }
        public double DataPoint2 { get; set; }
        public double DataPoint3 { get; set; }
        public double DataPoint4 { get; set; }
        public double DataPoint5 { get; set; }
        public double DataPoint6 { get; set; }
        public double DataPoint7 { get; set; }
        public double DataPoint8 { get; set; }
        public double DataPoint9 { get; set; }
        public double DataPoint10 { get; set; }
        public double DataPoint11 { get; set; }
        public double DataPoint12 { get; set; }
        public double DataPoint13 { get; set; }
        public double DataPoint14 { get; set; }
        public double DataPoint15 { get; set; }
        public double DataPoint16 { get; set; }
        public double DataPoint17 { get; set; }
        public double DataPoint18 { get; set; }
        public double DataPoint19 { get; set; }
        public double DataPoint20 { get; set; }
        public double DataPoint21 { get; set; }
        public double DataPoint22 { get; set; }
        public double DataPoint23 { get; set; }
        public double DataPoint24 { get; set; }
        public double DataPoint25 { get; set; }
        public double DataPoint26 { get; set; }
        public double DataPoint27 { get; set; }
        public double DataPoint28 { get; set; }


        public static BankTransaction For(string[] orderedValues)
        {
            throw new NotImplementedException(); 
        }
    }
}