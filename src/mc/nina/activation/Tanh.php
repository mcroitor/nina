<?php

namespace mc\nina\activation;

/**
 * Tanh (Hyperbolic Tangent) activation function implementation
 * 
 * The tanh function is defined as: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * Or equivalently: f(x) = tanh(x)
 * 
 * Properties:
 * - Output range: (-1, 1)
 * - Smooth gradient
 * - S-shaped curve (similar to sigmoid but centered at 0)
 * - Zero-centered output (unlike sigmoid)
 * - Better for hidden layers than sigmoid
 * - Still can suffer from vanishing gradient problem
 */
class Tanh implements ActivationFunction
{
    /**
     * Maximum input value to prevent overflow
     */
    private const MAX_INPUT = 500.0;

    /**
     * Apply the tanh activation function
     * 
     * Formula: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
     * 
     * @param float $x The input value
     * @return float The output value between -1 and 1
     */
    public function activate(float $x): float
    {
        // Prevent overflow for very large negative values
        if ($x < -self::MAX_INPUT) {
            return -1.0;
        }
        
        // Prevent overflow for very large positive values  
        if ($x > self::MAX_INPUT) {
            return 1.0;
        }
        
        return tanh($x);
    }

    /**
     * Calculate the derivative of the tanh function
     * 
     * Formula: f'(x) = 1 - tanh²(x) = 1 - f(x)²
     * Where f(x) is the tanh activation of x
     * 
     * @param float $x The input value (same as used in activate())
     * @return float The derivative value
     */
    public function derivative(float $x): float
    {
        $tanh = $this->activate($x);
        return 1.0 - ($tanh * $tanh);
    }

    /**
     * Get the name of this activation function
     * 
     * @return string The name "tanh"
     */
    public function getName(): string
    {
        return "tanh";
    }

    /**
     * Get the output range of the tanh function
     * 
     * @return array An array containing [-1.0, 1.0] (exclusive bounds)
     */
    public function getRange(): array
    {
        return [-1.0, 1.0];
    }

    /**
     * Static factory method for convenience
     * 
     * @return Tanh A new instance of tanh activation
     */
    public static function create(): Tanh
    {
        return new self();
    }
}