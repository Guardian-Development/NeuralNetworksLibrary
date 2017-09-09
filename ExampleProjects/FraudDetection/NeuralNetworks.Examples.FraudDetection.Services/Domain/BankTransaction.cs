using System;
using System.Globalization;

namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    public enum BankTransactionClass
    {
        Legitimate,
        Fraudulent
    }

    public class BankTransaction
    {
        public BankTransactionClass Class { get; internal set; }

        public int TimeOfTransaction { get; internal set; }
        public double Amount { get; internal set; }

        public double DataPoint1 { get; internal set; }
        public double DataPoint2 { get; internal set; }
        public double DataPoint3 { get; internal set; }
        public double DataPoint4 { get; internal set; }
        public double DataPoint5 { get; internal set; }
        public double DataPoint6 { get; internal set; }
        public double DataPoint7 { get; internal set; }
        public double DataPoint8 { get; internal set; }
        public double DataPoint9 { get; internal set; }
        public double DataPoint10 { get; internal set; }
        public double DataPoint11 { get; internal set; }
        public double DataPoint12 { get; internal set; }
        public double DataPoint13 { get; internal set; }
        public double DataPoint14 { get; internal set; }
        public double DataPoint15 { get; internal set; }
        public double DataPoint16 { get; internal set; }
        public double DataPoint17 { get; internal set; }
        public double DataPoint18 { get; internal set; }
        public double DataPoint19 { get; internal set; }
        public double DataPoint20 { get; internal set; }
        public double DataPoint21 { get; internal set; }
        public double DataPoint22 { get; internal set; }
        public double DataPoint23 { get; internal set; }
        public double DataPoint24 { get; internal set; }
        public double DataPoint25 { get; internal set; }
        public double DataPoint26 { get; internal set; }
        public double DataPoint27 { get; internal set; }
        public double DataPoint28 { get; internal set; }


        public static BankTransaction For(string[] orderedValues)
        {
            ValidateOrderedValues(orderedValues); 

            return new BankTransaction
            {
                TimeOfTransaction = Int32.Parse(orderedValues[0], NumberStyles.AllowExponent),
                DataPoint1 = Double.Parse(orderedValues[1]),
                DataPoint2 = Double.Parse(orderedValues[2]),
                DataPoint3 = Double.Parse(orderedValues[3]),
                DataPoint4 = Double.Parse(orderedValues[4]),
                DataPoint5 = Double.Parse(orderedValues[5]),
                DataPoint6 = Double.Parse(orderedValues[6]),
                DataPoint7 = Double.Parse(orderedValues[7]),
                DataPoint8 = Double.Parse(orderedValues[8]),
                DataPoint9 = Double.Parse(orderedValues[9]),
                DataPoint10 = Double.Parse(orderedValues[10]),
                DataPoint11 = Double.Parse(orderedValues[11]),
                DataPoint12 = Double.Parse(orderedValues[12]),
                DataPoint13 = Double.Parse(orderedValues[13]),
                DataPoint14 = Double.Parse(orderedValues[14]),
                DataPoint15 = Double.Parse(orderedValues[15]),
                DataPoint16 = Double.Parse(orderedValues[16]),
                DataPoint17 = Double.Parse(orderedValues[17]),
                DataPoint18 = Double.Parse(orderedValues[18]),
                DataPoint19 = Double.Parse(orderedValues[19]),
                DataPoint20 = Double.Parse(orderedValues[20]),
                DataPoint21 = Double.Parse(orderedValues[21]),
                DataPoint22 = Double.Parse(orderedValues[22]),
                DataPoint23 = Double.Parse(orderedValues[23]),
                DataPoint24 = Double.Parse(orderedValues[24]),
                DataPoint25 = Double.Parse(orderedValues[25]),
                DataPoint26 = Double.Parse(orderedValues[26]),
                DataPoint27 = Double.Parse(orderedValues[27]),
                DataPoint28 = Double.Parse(orderedValues[28]),
                Amount = Double.Parse(orderedValues[29]),
                Class = orderedValues[30].Contains("1") ? 
                    BankTransactionClass.Fraudulent : BankTransactionClass.Legitimate
            };
        }

        private static void ValidateOrderedValues(string[] orderedValues)
        {
            if(orderedValues.Length != 31)
            {
                throw new ArgumentException("Each Bank Transaction should have 31 values"); 
            }
        }
    }
}