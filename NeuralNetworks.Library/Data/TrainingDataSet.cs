namespace NeuralNetworks.Library.Data
{
    public sealed class TrainingDataSet
    {
        public double[] Inputs { get; }
        public double[] Outputs { get; }

        private TrainingDataSet(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public static TrainingDataSet For(double[] inputs, double[] outputs)
            =>  new TrainingDataSet(inputs, outputs);
    }
}
