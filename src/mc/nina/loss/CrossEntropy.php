<?php

namespace mc\nina\loss;

use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Cross-Entropy loss function implementation
 * 
 * Cross-entropy is the standard loss function for classification tasks.
 * It measures the difference between predicted probability distributions and true labels.
 * 
 * Formula: CE = -Σ(target[i] * log(predicted[i]))
 * 
 * Properties:
 * - Always non-negative (>= 0)
 * - Heavily penalizes confident wrong predictions
 * - Works well with softmax activation in output layer
 * - Provides strong gradients for optimization
 * 
 * @package mc\nina\loss
 * @author Mihail Croitor
 * @version 1.0
 * @since 0.3.0
 */
class CrossEntropy implements LossFunction
{
    /**
     * Small epsilon value to prevent log(0) which would cause -inf
     */
    private const EPSILON = 1e-15;

    /**
     * Calculate Cross-Entropy loss between predicted and target values
     * 
     * Computes CE = -Σ(target[i] * log(predicted[i] + ε))
     * where ε is a small value to prevent log(0).
     * 
     * @param array<float> $predicted Array of predicted probabilities (should sum to 1)
     * @param array<float> $target Array of target probabilities or one-hot encoded labels
     * 
     * @return float The cross-entropy loss value (>= 0)
     * 
     * @throws ValidationException When arrays have different sizes or are empty
     * @throws ComputationException When arrays contain invalid values
     * 
     * @example
     * ```php
     * $ce = new CrossEntropy();
     * $predicted = [0.7, 0.2, 0.1]; // Predicted probabilities
     * $target = [1.0, 0.0, 0.0];    // True class is first one
     * $loss = $ce->calculate($predicted, $target); // Returns: 0.357
     * ```
     */
    public function calculate(array $predicted, array $target): float
    {
        $this->validateInputs($predicted, $target);
        
        $loss = 0.0;
        
        for ($i = 0; $i < count($predicted); $i++) {
            // Clip predicted values to prevent log(0)
            $clipped_pred = max(self::EPSILON, min(1.0 - self::EPSILON, $predicted[$i]));
            
            // Only add to loss if target has non-zero probability
            if ($target[$i] > 0) {
                $loss -= $target[$i] * log($clipped_pred);
            }
        }
        
        // Validate result
        if (!is_finite($loss) || $loss < 0) {
            throw ComputationException::invalidNumericalValue(
                $loss,
                'Cross-entropy calculation result'
            );
        }
        
        return $loss;
    }

    /**
     * Calculate the derivative of Cross-Entropy with respect to predicted values
     * 
     * The derivative of Cross-Entropy is: ∂CE/∂predicted[i] = -target[i] / predicted[i]
     * 
     * @param array<float> $predicted Array of predicted probabilities
     * @param array<float> $target Array of target probabilities
     * 
     * @return array<float> Array of partial derivatives
     * 
     * @throws ValidationException When arrays have different sizes or are empty
     * @throws ComputationException When arrays contain invalid values
     * 
     * @example
     * ```php
     * $ce = new CrossEntropy();
     * $predicted = [0.8, 0.2];
     * $target = [1.0, 0.0];
     * $gradients = $ce->derivative($predicted, $target); // Returns: [-1.25, 0.0]
     * ```
     */
    public function derivative(array $predicted, array $target): array
    {
        $this->validateInputs($predicted, $target);
        
        $gradients = [];
        
        for ($i = 0; $i < count($predicted); $i++) {
            // Clip predicted values to prevent division by 0
            $clipped_pred = max(self::EPSILON, min(1.0 - self::EPSILON, $predicted[$i]));
            
            // Calculate gradient: -target[i] / predicted[i]
            $gradient = -$target[$i] / $clipped_pred;
            
            // Validate gradient
            if (!is_finite($gradient)) {
                throw ComputationException::invalidNumericalValue(
                    $gradient,
                    "Cross-entropy gradient at index {$i}"
                );
            }
            
            $gradients[] = $gradient;
        }
        
        return $gradients;
    }

    /**
     * Get the name of this loss function
     * 
     * @return string The name "cross_entropy"
     */
    public function getName(): string
    {
        return "cross_entropy";
    }

    /**
     * Get information about the Cross-Entropy loss function
     * 
     * @return array<string, mixed> Comprehensive information about Cross-Entropy
     */
    public function getInfo(): array
    {
        return [
            'name' => 'Cross-Entropy',
            'abbreviation' => 'CE',
            'range' => [0.0, PHP_FLOAT_MAX],
            'use_cases' => [
                'Multi-class classification',
                'Binary classification',
                'Probability distribution matching',
                'When outputs are probabilities'
            ],
            'properties' => [
                'Always non-negative',
                'Convex function for fixed targets',
                'Heavily penalizes confident wrong predictions',
                'Works well with softmax activation',
                'Provides strong gradients away from correct answers'
            ],
            'formula' => 'CE = -Σ(target[i] * log(predicted[i] + ε))',
            'derivative_formula' => '∂CE/∂predicted[i] = -target[i] / predicted[i]',
            'epsilon' => self::EPSILON
        ];
    }

    /**
     * Static factory method for convenience
     * 
     * @return CrossEntropy A new instance of Cross-Entropy loss function
     */
    public static function create(): CrossEntropy
    {
        return new self();
    }

    /**
     * Calculate categorical cross-entropy for one-hot encoded labels
     * 
     * Specialized method for the common case where target is one-hot encoded
     * (only one element is 1.0, rest are 0.0).
     * 
     * @param array<float> $predicted Array of predicted probabilities
     * @param int $true_class_index Index of the true class (0-based)
     * 
     * @return float The cross-entropy loss value
     * 
     * @example
     * ```php
     * $ce = new CrossEntropy();
     * $predicted = [0.1, 0.8, 0.1]; // Predicted probabilities for 3 classes
     * $true_class = 0; // True class is the first one
     * $loss = $ce->categoricalCrossEntropy($predicted, $true_class);
     * ```
     */
    public function categoricalCrossEntropy(array $predicted, int $true_class_index): float
    {
        if (empty($predicted)) {
            throw ValidationException::invalidValue('predicted', 'empty array', 'non-empty array');
        }
        
        if ($true_class_index < 0 || $true_class_index >= count($predicted)) {
            throw ValidationException::outOfRange(
                'true_class_index',
                $true_class_index,
                0,
                count($predicted) - 1
            );
        }
        
        // Clip predicted value to prevent log(0)
        $clipped_pred = max(self::EPSILON, min(1.0 - self::EPSILON, $predicted[$true_class_index]));
        
        return -log($clipped_pred);
    }

    /**
     * Validate input arrays for loss calculation
     * 
     * @param array<float> $predicted Array of predicted values
     * @param array<float> $target Array of target values
     * 
     * @throws ValidationException When validation fails
     * @throws ComputationException When arrays contain invalid values
     */
    private function validateInputs(array $predicted, array $target): void
    {
        // Check if arrays are empty
        if (empty($predicted)) {
            throw ValidationException::invalidValue('predicted', 'empty array', 'non-empty array');
        }
        
        if (empty($target)) {
            throw ValidationException::invalidValue('target', 'empty array', 'non-empty array');
        }
        
        // Check array sizes match
        if (count($predicted) !== count($target)) {
            throw ValidationException::invalidArraySize(
                'predicted',
                count($predicted),
                count($target)
            );
        }
        
        // Validate predicted values (should be probabilities)
        foreach ($predicted as $index => $value) {
            if (!is_numeric($value)) {
                throw ValidationException::invalidValue(
                    "predicted[{$index}]",
                    $value,
                    'numeric',
                    "got " . gettype($value)
                );
            }
            
            $value = (float)$value;
            
            if (!is_finite($value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "predicted value at index {$index}"
                );
            }
            
            if ($value < 0.0 || $value > 1.0) {
                throw ValidationException::outOfRange(
                    "predicted[{$index}]",
                    $value,
                    0.0,
                    1.0
                );
            }
        }
        
        // Validate target values (should be probabilities)
        foreach ($target as $index => $value) {
            if (!is_numeric($value)) {
                throw ValidationException::invalidValue(
                    "target[{$index}]",
                    $value,
                    'numeric',
                    "got " . gettype($value)
                );
            }
            
            $value = (float)$value;
            
            if (!is_finite($value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "target value at index {$index}"
                );
            }
            
            if ($value < 0.0 || $value > 1.0) {
                throw ValidationException::outOfRange(
                    "target[{$index}]",
                    $value,
                    0.0,
                    1.0
                );
            }
        }
    }
}