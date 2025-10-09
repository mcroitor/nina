<?php

namespace mc\nina\activation;

/**
 * Softmax activation function implementation
 * 
 * The softmax function is defined as: f(x_i) = e^(x_i) / Σ(e^(x_j)) for all j
 * 
 * Properties:
 * - Output range: (0, 1) for each element
 * - Sum of all outputs equals 1 (probability distribution)
 * - Used for multi-class classification output layer
 * - Emphasizes the largest input values
 * - Differentiable everywhere
 * 
 * Note: Softmax is typically applied to vectors, not single values.
 * This implementation provides both single-value and vector interfaces.
 */
class Softmax implements ActivationFunction
{
    /**
     * Maximum input value to prevent overflow
     */
    private const MAX_INPUT = 500.0;

    /**
     * Apply the softmax activation function to a single value
     * 
     * Note: For proper softmax behavior, use activateVector() method
     * This method assumes the input is part of a vector with single element
     * 
     * @param float $x The input value
     * @return float The output value (always 1.0 for single element)
     */
    public function activate(float $x): float
    {
        // For single element, softmax always returns 1.0
        // because e^x / e^x = 1
        return 1.0;
    }

    /**
     * Apply the softmax activation function to a vector
     * 
     * Formula: f(x_i) = e^(x_i - max(x)) / Σ(e^(x_j - max(x)))
     * We subtract max(x) for numerical stability
     * 
     * @param array $vector The input vector
     * @return array The output vector (probability distribution)
     */
    public function activateVector(array $vector): array
    {
        if (empty($vector)) {
            return [];
        }

        // Find maximum value for numerical stability
        $max = max($vector);
        
        // Prevent overflow by clamping max value
        $max = min($max, self::MAX_INPUT);
        
        // Calculate exponentials with shifted values
        $exponentials = [];
        $sum = 0.0;
        
        foreach ($vector as $value) {
            $shiftedValue = $value - $max;
            
            // Prevent underflow for very negative values
            if ($shiftedValue < -self::MAX_INPUT) {
                $exp = 0.0;
            } else {
                $exp = exp($shiftedValue);
            }
            
            $exponentials[] = $exp;
            $sum += $exp;
        }
        
        // Avoid division by zero
        if ($sum == 0.0) {
            // Return uniform distribution
            $uniformProb = 1.0 / count($vector);
            return array_fill(0, count($vector), $uniformProb);
        }
        
        // Normalize to get probabilities
        $result = [];
        foreach ($exponentials as $exp) {
            $result[] = $exp / $sum;
        }
        
        return $result;
    }

    /**
     * Calculate the derivative of the softmax function
     * 
     * Note: Softmax derivative is complex and depends on the entire vector
     * For single value, this is a simplified approximation
     * 
     * @param float $x The input value
     * @return float The derivative value (simplified)
     */
    public function derivative(float $x): float
    {
        // Simplified derivative for single value
        // For proper softmax gradient, use derivativeVector() method
        $softmax = $this->activate($x);
        return $softmax * (1.0 - $softmax);
    }

    /**
     * Calculate the Jacobian matrix of softmax function
     * 
     * Formula: ∂f_i/∂x_j = f_i * (δ_ij - f_j)
     * where δ_ij is the Kronecker delta (1 if i=j, 0 otherwise)
     * 
     * @param array $vector The input vector
     * @return array 2D array representing the Jacobian matrix
     */
    public function derivativeVector(array $vector): array
    {
        $softmaxOutput = $this->activateVector($vector);
        $size = count($softmaxOutput);
        $jacobian = [];
        
        for ($i = 0; $i < $size; $i++) {
            $jacobian[$i] = [];
            for ($j = 0; $j < $size; $j++) {
                if ($i == $j) {
                    // Diagonal elements: f_i * (1 - f_i)
                    $jacobian[$i][$j] = $softmaxOutput[$i] * (1.0 - $softmaxOutput[$i]);
                } else {
                    // Off-diagonal elements: -f_i * f_j
                    $jacobian[$i][$j] = -$softmaxOutput[$i] * $softmaxOutput[$j];
                }
            }
        }
        
        return $jacobian;
    }

    /**
     * Get the name of this activation function
     * 
     * @return string The name "softmax"
     */
    public function getName(): string
    {
        return "softmax";
    }

    /**
     * Get the output range of the softmax function
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
     * @return Softmax A new instance of softmax activation
     */
    public static function create(): Softmax
    {
        return new self();
    }
}