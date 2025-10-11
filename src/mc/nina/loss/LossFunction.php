<?php

namespace mc\nina\loss;

/**
 * Interface for loss functions used in neural network training
 * 
 * Loss functions measure the difference between predicted and actual outputs.
 * They are essential for backpropagation algorithm as they provide the initial
 * gradient for the backward pass through the network.
 * 
 * @package mc\nina\loss
 * @author Mihail Croitor
 * @version 1.0
 * @since 0.3.0
 */
interface LossFunction
{
    /**
     * Calculate the loss between predicted and target values
     * 
     * Computes the loss value that quantifies how far the predictions
     * are from the true target values. Lower values indicate better performance.
     * 
     * @param array<float> $predicted Array of predicted values from the neural network
     * @param array<float> $target Array of target (true) values
     * 
     * @return float The calculated loss value (always >= 0)
     * 
     * @throws \InvalidArgumentException When arrays have different sizes or invalid values
     */
    public function calculate(array $predicted, array $target): float;

    /**
     * Calculate the derivative of the loss function with respect to predictions
     * 
     * Computes the gradient of the loss function, which is used as the starting point
     * for backpropagation. The derivative indicates how much the loss would change
     * if each predicted value was increased by a small amount.
     * 
     * @param array<float> $predicted Array of predicted values from the neural network
     * @param array<float> $target Array of target (true) values
     * 
     * @return array<float> Array of partial derivatives ∂Loss/∂predicted[i]
     * 
     * @throws \InvalidArgumentException When arrays have different sizes or invalid values
     */
    public function derivative(array $predicted, array $target): array;

    /**
     * Get the name of the loss function
     * 
     * @return string The name of the loss function (e.g., "mse", "cross_entropy")
     */
    public function getName(): string;

    /**
     * Get information about the loss function
     * 
     * Returns metadata about the loss function including its typical use cases,
     * output range, and mathematical properties.
     * 
     * @return array<string, mixed> Information about the loss function
     * [
     *   'name' => string,
     *   'range' => [min_value, max_value],
     *   'use_cases' => array<string>,
     *   'properties' => array<string>
     * ]
     */
    public function getInfo(): array;
}