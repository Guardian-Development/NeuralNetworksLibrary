namespace NeuralNetworks.Tests.IntegrationTests.Training.BackPropagationTests
{
    public abstract class Assertor<T> where T : class
    {
        protected T ExpectedItem;

        protected Assertor(T expectedItem)
        {
            ExpectedItem = expectedItem;
        }

        public abstract void Assert(T actualItem);
    }
}
