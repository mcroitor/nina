<?php

namespace mc\nina\optimizer;

use mc\nina\Cell;

/**
 * Interface for neural network weight optimizers.
 * 
 * Defines the contract for optimization algorithms (SGD, Adam, RMSprop)
 * that determine how weights are updated based on computed gradients.
 * 
 * @package mc\nina\optimizer
 * @author Mihail Croitor
 * @version 1.0.0
 */
interface Optimizer
{
    /**
     * Get the name of the optimizer
     * 
     * @return string Optimizer identifier (e.g., 'sgd', 'adam', 'rmsprop')
     */
    public function getName(): string;

    /**
     * Update weights of a cell given the computed gradients and learning rate
     * 
     * @param Cell $cell The neuron whose weights should be updated
     * @param array<float> $gradients Gradient vector for the weights
     * @param float $learningRate Current effective learning rate
     * 
     * @throws \mc\nina\exception\ValidationException If parameters or sizes are invalid
     * @throws \mc\nina\exception\ComputationException If numerical errors occur
     */
    public function update(Cell $cell, array $gradients, float $learningRate): void;

    /**
     * Reset internal optimizer state (e.g. velocities, moving averages, step counts)
     */
    public function reset(): void;

    /**
     * Set L2 weight decay regularization coefficient
     * 
     * @param float $weightDecay Non-negative weight decay coefficient
     */
    public function setWeightDecay(float $weightDecay): void;

    /**
     * Get L2 weight decay regularization coefficient
     * 
     * @return float
     */
    public function getWeightDecay(): float;

    /**
     * Set L1 regularization coefficient (Lasso)
     * 
     * @param float $l1 Non-negative L1 coefficient
     */
    public function setL1(float $l1): void;

    /**
     * Get L1 regularization coefficient
     * 
     * @return float
     */
    public function getL1(): float;
}
