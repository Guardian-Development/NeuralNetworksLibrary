namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public sealed class BackPropagationTester
    {
        //TODO: reimplement this class using assertor pattern
        private readonly double learningRate;
        private readonly double momentum;

        private BackPropagationTester(double learningRate, double momentum)
        {
            this.momentum = momentum;
            this.learningRate = learningRate;
        }

        public static BackPropagationTester For(double learningRate,double momentum)
        {
            return new BackPropagationTester(learningRate, momentum);
        }
    }
}