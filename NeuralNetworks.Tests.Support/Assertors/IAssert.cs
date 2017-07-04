namespace NeuralNetworks.Tests.Support.Assertors
{
    public interface IAssert<T>
    {
        void Assert(T actualItem);
    }

    public interface IAssertBuilder<T>
    {
        IAssert<T> Build(); 
    }
}
