namespace NeuralNetworks.Tests.Support.Assertors
{
    public interface IAssert<in T>
    {
        void Assert(T actualItem);
    }

    public interface IAssertBuilder<in T>
    {
        IAssert<T> Build(); 
    }
}
