namespace NeuralNetworks.Tests.Support.Assertors
{
    public class FieldAssertor<T> : IAssert<T>
    {
		public static FieldAssertor<T> NoAssert => new FieldAssertor<T>();
		
        private readonly IAssert<T> assertor;

        public FieldAssertor(IAssert<T> assertor)
        {
            this.assertor = assertor;
        }
        
        public FieldAssertor() 
        {}

        public void Assert(T actualItem)
        {
            if(assertor != null)
            {
                assertor.Assert(actualItem);
            }
        }

        public static FieldAssertor<T> Create(IAssert<T> assertor)
            => new FieldAssertor<T>(assertor);
    }
}
