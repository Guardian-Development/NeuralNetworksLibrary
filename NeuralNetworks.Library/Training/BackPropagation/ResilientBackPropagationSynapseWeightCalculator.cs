using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using NeuralNetworks.Library.Components;
using NeuralNetworks.Library.Extensions;
using NeuralNetworks.Library.Logging;

namespace NeuralNetworks.Library.Training.BackPropagation
{
    public sealed class ResilientBackPropagationSynapseWeightCalculator : IUpdateSynapseWeights
    {
        private static ILogger Log => LoggerProvider.For<ResilientBackPropagationSynapseWeightCalculator>();

        private Dictionary<Synapse, double> synapseToPreviousErrorRate 
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
                    errorGradientSign);

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
        private const double negativeWeightUpdateAmount = 0.5; 
        private const double positiveWeightUpdateAmount = 1.2;
        private const double maximumWeightUpdate = 50.0; 
        private const double minimumWeightUpdate = 1e-6;
        private const double initialUpdateValue = 0.1;

        public static int SignOfSum(params double[] values) 
        {
            if(values.Contains(0))
            {
                return 0; 
            }
            return Math.Sign(values.Sum()); 
        }
        
        public static void ContinueToIncreaseWeightTowardsErrorGradientMinimum(
            Synapse synapse,
            double currentErrorGradient)
        {
            var newWeightDelta = Math.Min(synapse.WeightDelta * positiveWeightUpdateAmount, maximumWeightUpdate);

            UpdateSynapseWeightCalculatingWeightDeltaDirection(
                synapse,
                currentErrorGradient,
                newWeightDelta); 
        }

        public static void RevertPreviousWeightAdjustment(Synapse synapse)
        {
            var newWeightDelta = Math.Max(synapse.WeightDelta * negativeWeightUpdateAmount, minimumWeightUpdate); 
            synapse.Weight = synapse.Weight - synapse.WeightDelta;
            synapse.WeightDelta = newWeightDelta;
        }

        public static void UpdateWeightWithoutCalculatingWeightDelta(
            Synapse synapse,
            double currentErrorGradient)
        {
            var weightDelta = synapse.WeightDelta == 0 ? initialUpdateValue : synapse.WeightDelta;

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
