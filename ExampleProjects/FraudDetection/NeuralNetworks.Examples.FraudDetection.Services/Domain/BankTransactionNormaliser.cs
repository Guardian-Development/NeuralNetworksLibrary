namespace NeuralNetworks.Examples.FraudDetection.Services.Domain
{
    internal static class BankTransactionNormaliser
    {
        public static BankTransaction NormaliseTransactionAmount(
            BankTransaction bankTransaction,
            double min, 
            double max)
        {
            return new BankTransaction()
            {
                TimeOfTransaction = bankTransaction.TimeOfTransaction,
                DataPoint1 = bankTransaction.DataPoint1,
                DataPoint2 = bankTransaction.DataPoint2,
                DataPoint3 = bankTransaction.DataPoint3,
                DataPoint4 = bankTransaction.DataPoint4,
                DataPoint5 = bankTransaction.DataPoint5,
                DataPoint6 = bankTransaction.DataPoint6,
                DataPoint7 = bankTransaction.DataPoint7,
                DataPoint8 = bankTransaction.DataPoint8,
                DataPoint9 = bankTransaction.DataPoint9,
                DataPoint10 = bankTransaction.DataPoint10,
                DataPoint11 = bankTransaction.DataPoint11,
                DataPoint12 = bankTransaction.DataPoint12,
                DataPoint13 = bankTransaction.DataPoint13,
                DataPoint14 = bankTransaction.DataPoint14,
                DataPoint15 = bankTransaction.DataPoint15,
                DataPoint16 = bankTransaction.DataPoint16,
                DataPoint17 = bankTransaction.DataPoint17,
                DataPoint18 = bankTransaction.DataPoint18,
                DataPoint19 = bankTransaction.DataPoint19,
                DataPoint20 = bankTransaction.DataPoint20,
                DataPoint21 = bankTransaction.DataPoint21,
                DataPoint22 = bankTransaction.DataPoint22,
                DataPoint23 = bankTransaction.DataPoint23,
                DataPoint24 = bankTransaction.DataPoint24,
                DataPoint25 = bankTransaction.DataPoint25,
                DataPoint26 = bankTransaction.DataPoint26,
                DataPoint27 = bankTransaction.DataPoint27,
                DataPoint28 = bankTransaction.DataPoint28,
                Amount = bankTransaction.Amount.NormaliseValue(min, max),
                Class = bankTransaction.Class
            };
        }

        private static double NormaliseValue(this double value, double min, double max)
        {
            return (value - min) / (max - min); 
        }
    }
}