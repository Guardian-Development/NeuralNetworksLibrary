namespace NeuralNetworks.Library.Components.Activation.Functions
{
    public interface IProvideNeuronActivation
    {
        double Activate(double sumOfWeights);
        double Derivative(double sumOfWeights); 
    }
}
