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
        public BankTransactionCategory Class { get; private set; }

        public int TimeOfTransaction { get; private set; }
        public double Amount { get; private set; }

        public double DataPoint1 { get; private set; }
        public double DataPoint2 { get; private set; }
        public double DataPoint3 { get; private set; }
        public double DataPoint4 { get; private set; }
        public double DataPoint5 { get; private set; }
        public double DataPoint6 { get; private set; }
        public double DataPoint7 { get; private set; }
        public double DataPoint8 { get; private set; }
        public double DataPoint9 { get; private set; }
        public double DataPoint10 { get; private set; }
        public double DataPoint11 { get; private set; }
        public double DataPoint12 { get; private set; }
        public double DataPoint13 { get; private set; }
        public double DataPoint14 { get; private set; }
        public double DataPoint15 { get; private set; }
        public double DataPoint16 { get; private set; }
        public double DataPoint17 { get; private set; }
        public double DataPoint18 { get; private set; }
        public double DataPoint19 { get; private set; }
        public double DataPoint20 { get; private set; }
        public double DataPoint21 { get; private set; }
        public double DataPoint22 { get; private set; }
        public double DataPoint23 { get; private set; }
        public double DataPoint24 { get; private set; }
        public double DataPoint25 { get; private set; }
        public double DataPoint26 { get; private set; }
        public double DataPoint27 { get; private set; }
        public double DataPoint28 { get; private set; }


        public static BankTransaction For(string[] orderedValues)
        {
            throw new NotImplementedException(); 
        }
    }
}