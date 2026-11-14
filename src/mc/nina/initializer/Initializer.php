<?php

namespace mc\nina\initializer;

/**
 * Interface for neural network weight initializers.
 * 
 * Provides strategies for initializing weights in neural network layers
 * to ensure optimal signal and gradient flow during backpropagation.
 * 
 * @package mc\nina\initializer
 * @author Mihail Croitor
 * @version 1.0.0
 */
interface Initializer
{
    /**
     * Generate an array of initialized weights
     * 
     * @param int $size Number of weights to generate (fan_in for the cell)
     * @param int $fanIn Number of inputs to the layer (default = size)
     * @param int $fanOut Number of neurons in the layer (default = size)
     * 
     * @return array<float> Array of initialized weight values
     */
    public function initialize(int $size, int $fanIn = 0, int $fanOut = 0): array;

    /**
     * Returns the name identifier of this initializer
     * 
     * @return string
     */
    public function getName(): string;
}
