<?php

namespace mc\nina\loss;

use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Mean Squared Error (MSE) loss function implementation
 * 
 * MSE is one of the most common loss functions for regression tasks.
 * It measures the average squared differences between predicted and target values.
 * 
 * Formula: MSE = (1/n) * Σ(predicted[i] - target[i])²
 * 
 * Properties:
 * - Always non-negative (>= 0)
 * - Heavily penalizes large errors due to squaring
 * - Smooth gradient for optimization
 * - Units are squared units of the target variable
 * 
 * @package mc\nina\loss
 * @author Mihail Croitor
 * @version 1.0
 * @since 0.3.0
 */
class MeanSquaredError implements LossFunction
{
    /**
     * Calculate Mean Squared Error between predicted and target values
     * 
     * Computes MSE = (1/n) * Σ(predicted[i] - target[i])²
     * 
     * @param array<float> $predicted Array of predicted values
     * @param array<float> $target Array of target values
     * 
     * @return float The MSE loss value (>= 0)
     * 
     * @throws ValidationException When arrays have different sizes or are empty
     * @throws ComputationException When arrays contain non-finite values
     * 
     * @example
     * ```php
     * $mse = new MeanSquaredError();
     * $predicted = [0.8, 0.3, 0.9];
     * $target = [1.0, 0.0, 1.0];
     * $loss = $mse->calculate($predicted, $target); // Returns: 0.0167
     * ```
     */
    public function calculate(array $predicted, array $target): float
    {
        $this->validateInputs($predicted, $target);
        
        $n = count($predicted);
        $sumSquaredErrors = 0.0;
        
        for ($i = 0; $i < $n; $i++) {
            $error = $predicted[$i] - $target[$i];
            $sumSquaredErrors += $error * $error;
        }
        
        $mse = $sumSquaredErrors / $n;
        
        // Validate result
        if (!is_finite($mse)) {
            throw ComputationException::invalidNumericalValue(
                $mse,
                'MSE calculation result'
            );
        }
        
        return $mse;
    }

    /**
     * Calculate the derivative of MSE with respect to predicted values
     * 
     * The derivative of MSE is: ∂MSE/∂predicted[i] = (2/n) * (predicted[i] - target[i])
     * 
     * @param array<float> $predicted Array of predicted values
     * @param array<float> $target Array of target values
     * 
     * @return array<float> Array of partial derivatives
     * 
     * @throws ValidationException When arrays have different sizes or are empty
     * @throws ComputationException When arrays contain non-finite values
     * 
     * @example
     * ```php
     * $mse = new MeanSquaredError();
     * $predicted = [0.8, 0.3];
     * $target = [1.0, 0.0];
     * $gradients = $mse->derivative($predicted, $target); // Returns: [-0.4, 0.6]
     * ```
     */
    public function derivative(array $predicted, array $target): array
    {
        $this->validateInputs($predicted, $target);
        
        $n = count($predicted);
        $gradients = [];
        
        for ($i = 0; $i < $n; $i++) {
            $gradient = (2.0 / $n) * ($predicted[$i] - $target[$i]);
            
            // Validate gradient
            if (!is_finite($gradient)) {
                throw ComputationException::invalidNumericalValue(
                    $gradient,
                    "MSE gradient at index {$i}"
                );
            }
            
            $gradients[] = $gradient;
        }
        
        return $gradients;
    }

    /**
     * Get the name of this loss function
     * 
     * @return string The name "mse"
     */
    public function getName(): string
    {
        return "mse";
    }

    /**
     * Get information about the MSE loss function
     * 
     * @return array<string, mixed> Comprehensive information about MSE
     */
    public function getInfo(): array
    {
        return [
            'name' => 'Mean Squared Error',
            'abbreviation' => 'MSE',
            'range' => [0.0, PHP_FLOAT_MAX],
            'use_cases' => [
                'Regression tasks',
                'Continuous output prediction',
                'When large errors should be heavily penalized'
            ],
            'properties' => [
                'Always non-negative',
                'Convex function (single global minimum)',
                'Smooth gradient everywhere',
                'Sensitive to outliers',
                'Units are squared units of target'
            ],
            'formula' => 'MSE = (1/n) * Σ(predicted[i] - target[i])²',
            'derivative_formula' => '∂MSE/∂predicted[i] = (2/n) * (predicted[i] - target[i])'
        ];
    }

    /**
     * Static factory method for convenience
     * 
     * @return MeanSquaredError A new instance of MSE loss function
     */
    public static function create(): MeanSquaredError
    {
        return new self();
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
        
        // Validate values
        foreach ($predicted as $index => $value) {
            if (!is_numeric($value)) {
                throw ValidationException::invalidValue(
                    "predicted[{$index}]",
                    $value,
                    'numeric',
                    "got " . gettype($value)
                );
            }
            
            if (!is_finite((float)$value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "predicted value at index {$index}"
                );
            }
        }
        
        foreach ($target as $index => $value) {
            if (!is_numeric($value)) {
                throw ValidationException::invalidValue(
                    "target[{$index}]",
                    $value,
                    'numeric',
                    "got " . gettype($value)
                );
            }
            
            if (!is_finite((float)$value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "target value at index {$index}"
                );
            }
        }
    }
}