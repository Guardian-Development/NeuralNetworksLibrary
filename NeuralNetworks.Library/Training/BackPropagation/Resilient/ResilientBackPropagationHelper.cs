using System;
using System.Collections.Generic;
using NeuralNetworks.Library.Components;

namespace NeuralNetworks.Library.Training.BackPropagation.Resilient
{
    internal static class ResilientBackPropagationHelper
    {
        private const double NegativeWeightUpdateAmount = 0.5;
        private const double PositiveWeightUpdateAmount = 1.2;
        private const double MaximumWeightUpdate = 50.0;
        private const double MinimumWeightUpdate = 1.0E-6;
        private const double InitialUpdateValue = 0.1;

        private const double ZeroTolerance = 0.00000000000000001d;

        public static int Sign(double value)
        {
            if (Math.Abs(value) < ZeroTolerance)
            {
                return 0;
            }
            if (value > 0)
            {
                return 1;
            }
            return -1;
        }

        public static double CalculateDeltaToContinueTowardsErrorGraidentMinimum(
            Synapse synapse,
            double currentPartialDerivative,
            Dictionary<Synapse, double> synapseToPreviousDeltaWithoutDirection)
        {
            if (synapseToPreviousDeltaWithoutDirection.TryGetValue(synapse, out var previousUpdateValue))
            {
                var delta = Math.Min(previousUpdateValue * PositiveWeightUpdateAmount, MaximumWeightUpdate);
                synapseToPreviousDeltaWithoutDirection[synapse] = delta;
                return Sign(currentPartialDerivative) * delta;
            }

            throw new InvalidOperationException($"You cannot increase a prevous delta is none is present.");
        }

        public static double CalculateDeltaToRevertPreviousWeightAdjustment(
            Synapse synapse,
            double currentPartialDerivative,
            Dictionary<Synapse, double> synapseToPreviousDeltaWithoutDirection)
        {
            if (synapseToPreviousDeltaWithoutDirection.TryGetValue(synapse, out var previousUpdateValue))
            {
                var delta = Math.Max(previousUpdateValue * NegativeWeightUpdateAmount, MinimumWeightUpdate);
                synapseToPreviousDeltaWithoutDirection[synapse] = delta;
                return -synapse.WeightDelta;
            }

            throw new InvalidOperationException($"You cannot revert a previous change if none is known.");
        }

        public static double CalculateDeltaDirection(
            Synapse synapse,
            double currentPartialDerivative,
            Dictionary<Synapse, double> synapseToPreviousDeltaWithoutDirection)
        {
            if (synapseToPreviousDeltaWithoutDirection.TryGetValue(synapse, out var previousUpdateValue))
            {
                return Sign(currentPartialDerivative) * previousUpdateValue;
            }

            synapseToPreviousDeltaWithoutDirection.Add(synapse, InitialUpdateValue);

            return CalculateDeltaDirection(
                synapse,
                currentPartialDerivative,
                synapseToPreviousDeltaWithoutDirection);
        }
    }
}