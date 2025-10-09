<?php

namespace mc\nina\activation;

/**
 * Interface for activation functions used in neural networks
 * 
 * Activation functions are mathematical functions that determine the output
 * of a neural network node. They introduce non-linearity into the network,
 * allowing it to learn complex patterns.
 */
interface ActivationFunction
{
    /**
     * Apply the activation function to a given input value
     * 
     * @param float $x The input value to the activation function
     * @return float The output value after applying the activation function
     */
    public function activate(float $x): float;

    /**
     * Calculate the derivative of the activation function
     * 
     * This is used during backpropagation to calculate gradients.
     * The derivative is calculated with respect to the input value.
     * 
     * @param float $x The input value (same as used in activate())
     * @return float The derivative value at the given input
     */
    public function derivative(float $x): float;

    /**
     * Get the name of the activation function
     * 
     * @return string The name of the activation function (e.g., "sigmoid", "relu")
     */
    public function getName(): string;

    /**
     * Get the output range of the activation function
     * 
     * Returns an array with two elements: [min_value, max_value]
     * Use PHP_FLOAT_MIN and PHP_FLOAT_MAX for unbounded ranges.
     * 
     * @return array An array containing [min_output, max_output]
     */
    public function getRange(): array;
}