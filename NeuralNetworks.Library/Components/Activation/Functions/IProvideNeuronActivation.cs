namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public interface IProvideNeuronActivation
    {
        double Activate(double x);
        double Derivative(double x); 
    }
}
