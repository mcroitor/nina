<?php

namespace mc\nina\initializer;

use mc\nina\exception\ValidationException;

/**
 * Random Uniform weight initializer.
 * 
 * Initializes weights with uniform distribution in range [min, max].
 * 
 * @package mc\nina\initializer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class RandomUniformInitializer implements Initializer
{
    private float $min;
    private float $max;

    /**
     * @param float $min Lower bound (default -0.5)
     * @param float $max Upper bound (default 0.5)
     */
    public function __construct(float $min = -0.5, float $max = 0.5)
    {
        if ($min >= $max) {
            throw ValidationException::invalidValue('min', $min, "float less than max ({$max})");
        }
        $this->min = $min;
        $this->max = $max;
    }

    public function getName(): string
    {
        return 'random_uniform';
    }

    public function initialize(int $size, int $fanIn = 0, int $fanOut = 0): array
    {
        if ($size <= 0) {
            throw ValidationException::invalidValue('size', $size, 'positive integer');
        }

        $weights = [];
        $range = $this->max - $this->min;
        for ($i = 0; $i < $size; $i++) {
            $weights[] = $this->min + (mt_rand() / mt_getrandmax()) * $range;
        }
        return $weights;
    }
}
