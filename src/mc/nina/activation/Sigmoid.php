<?php

namespace mc\nina\activation;

/**
 * Sigmoid activation function implementation
 * 
 * The sigmoid function is defined as: f(x) = 1 / (1 + e^(-x))
 * 
 * Properties:
 * - Output range: (0, 1)
 * - Smooth gradient
 * - S-shaped curve
 * - Good for binary classification output layer
 * - Can suffer from vanishing gradient problem for very large or small inputs
 */
class Sigmoid implements ActivationFunction
{
    /**
     * Maximum input value to prevent overflow
     */
    private const MAX_INPUT = 500.0;

    /**
     * Apply the sigmoid activation function
     * 
     * Formula: f(x) = 1 / (1 + e^(-x))
     * 
     * @param float $x The input value
     * @return float The output value between 0 and 1
     */
    public function activate(float $x): float
    {
        // Prevent overflow for very large negative values
        if ($x < -self::MAX_INPUT) {
            return 0.0;
        }
        
        // Prevent overflow for very large positive values  
        if ($x > self::MAX_INPUT) {
            return 1.0;
        }
        
        return 1.0 / (1.0 + exp(-$x));
    }

    /**
     * Calculate the derivative of the sigmoid function
     * 
     * Formula: f'(x) = f(x) * (1 - f(x))
     * Where f(x) is the sigmoid activation of x
     * 
     * @param float $x The input value (same as used in activate())
     * @return float The derivative value
     */
    public function derivative(float $x): float
    {
        $sigmoid = $this->activate($x);
        return $sigmoid * (1.0 - $sigmoid);
    }

    /**
     * Get the name of this activation function
     * 
     * @return string The name "sigmoid"
     */
    public function getName(): string
    {
        return "sigmoid";
    }

    /**
     * Get the output range of the sigmoid function
     * 
     * @return array An array containing [0.0, 1.0] (exclusive bounds)
     */
    public function getRange(): array
    {
        return [0.0, 1.0];
    }

    /**
     * Static factory method for convenience
     * 
     * @return Sigmoid A new instance of sigmoid activation
     */
    public static function create(): Sigmoid
    {
        return new self();
    }
}