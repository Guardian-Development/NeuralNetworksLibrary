using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training.BackPropagation.Resilient
{
    public sealed partial class ResilientBackPropagationSynapseWeightCalculator : IUpdateSynapseWeights
    {
        private static ILogger Log => LoggerProvider.For<ResilientBackPropagationSynapseWeightCalculator>();

        private readonly Dictionary<Synapse, double> synapseToPreviousPartialDerivative
            = new Dictionary<Synapse, double>(SynapseEqualityComparer.Instance());

        private readonly Dictionary<Synapse, double> synapseToPreviousDeltaWithoutDirection
            = new Dictionary<Synapse, double>(SynapseEqualityComparer.Instance());

        private ResilientBackPropagationSynapseWeightCalculator()
        {}

        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron, ParallelOptions parallelOptions)
        {
            neuron.InputSynapses.ForEach(
                synapse =>
                {
                    var previousPartialDerivative = GetPreviousPartialDerivativeOfSynapse(synapse);
                    var currentPartialDerivative = synapse.OutputNeuron.ErrorGradient * synapse.InputNeuron.Output;

                    var weightDelta = CalculateSynapseWeightDelta(
                        synapse,
                        currentPartialDerivative,
                        previousPartialDerivative);

                    synapse.WeightDelta = weightDelta;
                    synapse.Weight = synapse.Weight + weightDelta;
                    Log.LogTrace($"{nameof(synapse.Weight)} : {synapse.Weight}");
                });
        }

        private double GetPreviousPartialDerivativeOfSynapse(Synapse synapse)
        {
            if (synapseToPreviousPartialDerivative.TryGetValue(synapse, out var previousPartialDerivative))
            {
                return previousPartialDerivative;
            }

            synapseToPreviousPartialDerivative[synapse] = 0;
            return 0;
        }

        private double CalculateSynapseWeightDelta(
            Synapse synapse,
            double currentPartialDerivative,
            double previousPartialDerivative)
        {
            var errorGradientSign =
                ResilientBackPropagationHelper.Sign(currentPartialDerivative * previousPartialDerivative);

            double weightDelta;

            if (errorGradientSign > 0)
            {
                Log.LogTrace($"{nameof(errorGradientSign)} greater than 0");

                weightDelta = ResilientBackPropagationHelper
                    .CalculateDeltaToContinueTowardsErrorGraidentMinimum(
                        synapse,
                        currentPartialDerivative,
                        synapseToPreviousDeltaWithoutDirection);

                synapseToPreviousPartialDerivative[synapse] = currentPartialDerivative;
            }
            else if (errorGradientSign < 0)
            {
                Log.LogTrace($"{nameof(errorGradientSign)} less than 0");
                weightDelta = ResilientBackPropagationHelper
                    .CalculateDeltaToRevertPreviousWeightAdjustment(
                        synapse,
                        currentPartialDerivative,
                        synapseToPreviousDeltaWithoutDirection);

                synapseToPreviousPartialDerivative[synapse] = 0; //0 so no adjustment next iteration. //TODO: refactor this. 
            }
            else
            {
                Log.LogTrace($"{nameof(errorGradientSign)} equal to 0");

                weightDelta = ResilientBackPropagationHelper
                    .CalculateDeltaDirection(
                        synapse,
                        currentPartialDerivative,
                        synapseToPreviousDeltaWithoutDirection);

                synapseToPreviousPartialDerivative[synapse] = currentPartialDerivative; 
            }

            Log.LogTrace($"{nameof(weightDelta)} : {weightDelta}");
            return weightDelta;
        }

        public static ResilientBackPropagationSynapseWeightCalculator Create()
            => new ResilientBackPropagationSynapseWeightCalculator();
    }
}