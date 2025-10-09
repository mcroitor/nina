<?php

namespace mc\nina\activation;

/**
 * ReLU (Rectified Linear Unit) activation function implementation
 * 
 * The ReLU function is defined as: f(x) = max(0, x)
 * 
 * Properties:
 * - Output range: [0, +∞)
 * - Simple and computationally efficient
 * - Non-saturating for positive inputs
 * - Helps mitigate vanishing gradient problem
 * - Can suffer from "dying ReLU" problem (neurons always output 0)
 * - Most popular activation for hidden layers in modern neural networks
 */
class ReLU implements ActivationFunction
{
    /**
     * Apply the ReLU activation function
     * 
     * Formula: f(x) = max(0, x)
     * 
     * @param float $x The input value
     * @return float The output value (0 if x < 0, x if x >= 0)
     */
    public function activate(float $x): float
    {
        return max(0.0, $x);
    }

    /**
     * Calculate the derivative of the ReLU function
     * 
     * Formula: f'(x) = 1 if x > 0, 0 if x <= 0
     * Note: Technically undefined at x = 0, but we return 0 by convention
     * 
     * @param float $x The input value (same as used in activate())
     * @return float The derivative value (0 or 1)
     */
    public function derivative(float $x): float
    {
        return $x > 0.0 ? 1.0 : 0.0;
    }

    /**
     * Get the name of this activation function
     * 
     * @return string The name "relu"
     */
    public function getName(): string
    {
        return "relu";
    }

    /**
     * Get the output range of the ReLU function
     * 
     * @return array An array containing [0.0, PHP_FLOAT_MAX] (unbounded positive)
     */
    public function getRange(): array
    {
        return [0.0, PHP_FLOAT_MAX];
    }

    /**
     * Static factory method for convenience
     * 
     * @return ReLU A new instance of ReLU activation
     */
    public static function create(): ReLU
    {
        return new self();
    }
}