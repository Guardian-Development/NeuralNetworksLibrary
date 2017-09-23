using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class ResilientBackPropagationSynapseWeightCalculator : IUpdateSynapseWeights
    {
        private static ILogger Log => LoggerProvider.For<ResilientBackPropagationSynapseWeightCalculator>();

        private readonly Dictionary<Synapse, double> synapseToPreviousErrorRate 
            = new Dictionary<Synapse, double>();

        private ResilientBackPropagationSynapseWeightCalculator()
        {}

        public void CalculateAndUpdateInputSynapseWeights(Neuron neuron, ParallelOptions parallelOptions)
        {
            neuron.InputSynapses.ForEach(
                    synapse => {
                        synapseToPreviousErrorRate.TryGetValue(synapse, out var previousErrorGraient);

                        UpdateSynapseWeight(
                            synapse,
                            neuron.ErrorGradient,
                            previousErrorGraient);
                    });
        }

        private void UpdateSynapseWeight(
            Synapse synapse, 
            double currentErrorGradient, 
            double previousErrorGraient)
        {
            var errorGradientSign = ResilientBackPropagationHelper.SignOfSum(
                currentErrorGradient, 
                previousErrorGraient);

            if(errorGradientSign > 0)
            {
                ResilientBackPropagationHelper.ContinueToIncreaseWeightTowardsErrorGradientMinimum(
                    synapse, 
                    currentErrorGradient);

                RecordLatestSeenErrorGradient(synapse, currentErrorGradient); 
            }
            else if(errorGradientSign < 0)
            {
                ResilientBackPropagationHelper.RevertPreviousWeightAdjustment(synapse);
                RecordLatestSeenErrorGradient(synapse, errorGradient: 0);
            }
            else if(errorGradientSign == 0)
            {
                ResilientBackPropagationHelper.UpdateWeightWithoutCalculatingWeightDelta(
                    synapse, 
                    currentErrorGradient);
                
                RecordLatestSeenErrorGradient(synapse, currentErrorGradient);
            }
        }

        private void RecordLatestSeenErrorGradient(Synapse synapse, double errorGradient)
        {
            synapseToPreviousErrorRate[synapse] = errorGradient; 
        }

        public static ResilientBackPropagationSynapseWeightCalculator Create()
            => new ResilientBackPropagationSynapseWeightCalculator();
    }

    internal static class ResilientBackPropagationHelper
    {
        private const double NegativeWeightUpdateAmount = 0.5; 
        private const double PositiveWeightUpdateAmount = 1.2;
        private const double MaximumWeightUpdate = 50.0; 
        private const double MinimumWeightUpdate = 1.0E-6;
        private const double InitialUpdateValue = 0.1;

        public static int SignOfSum(params double[] values) 
            => values.Contains(0) ? 0 : Math.Sign(values.Sum());

        public static void ContinueToIncreaseWeightTowardsErrorGradientMinimum(
            Synapse synapse,
            double currentErrorGradient)
        {
            var newDelta = synapse.WeightDelta * PositiveWeightUpdateAmount;
            var newWeightDelta = Math.Min(newDelta, MaximumWeightUpdate);

            UpdateSynapseWeightCalculatingWeightDeltaDirection(
                synapse,
                currentErrorGradient,
                newWeightDelta); 
        }

        public static void RevertPreviousWeightAdjustment(Synapse synapse)
        {
            var newDelta = synapse.WeightDelta * NegativeWeightUpdateAmount;
            var newWeightDelta = Math.Max(newDelta, MinimumWeightUpdate); 
            synapse.Weight = synapse.Weight - synapse.WeightDelta;
            synapse.WeightDelta = newWeightDelta;
        }

        public static void UpdateWeightWithoutCalculatingWeightDelta(
            Synapse synapse,
            double currentErrorGradient)
        {
            var weightDelta = synapse.WeightDelta == 0 ? InitialUpdateValue : synapse.WeightDelta;

            UpdateSynapseWeightCalculatingWeightDeltaDirection(
                synapse,
                currentErrorGradient,
                weightDelta);
        }

        private static void UpdateSynapseWeightCalculatingWeightDeltaDirection(
            Synapse synapse, 
            double currentErrorGradient, 
            double weightDelta)
        {
            var weightDeltaWithDirection = - SignOfSum(currentErrorGradient) * weightDelta; 
            synapse.WeightDelta = weightDeltaWithDirection;
            synapse.Weight = synapse.Weight + synapse.WeightDelta;
        }
    }
}
