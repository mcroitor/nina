<?php

namespace mc\nina\initializer;

use mc\nina\exception\ValidationException;

/**
 * He (Kaiming) weight initializer.
 * 
 * Optimized for networks using ReLU or LeakyReLU activation functions.
 * Accounts for half the activations being zeroed out by ReLU.
 * 
 * Reference: Kaiming He et al., 2015 -
 * "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
 * 
 * @package mc\nina\initializer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class HeInitializer implements Initializer
{
    private bool $uniform;

    /**
     * @param bool $uniform True for He Uniform, false for He Normal (default true)
     */
    public function __construct(bool $uniform = true)
    {
        $this->uniform = $uniform;
    }

    public function getName(): string
    {
        return $this->uniform ? 'he_uniform' : 'he_normal';
    }

    public function initialize(int $size, int $fanIn = 0, int $fanOut = 0): array
    {
        if ($size <= 0) {
            throw ValidationException::invalidValue('size', $size, 'positive integer');
        }

        $in = ($fanIn > 0) ? $fanIn : $size;
        $weights = [];

        if ($this->uniform) {
            // Uniform in [-limit, limit], where limit = sqrt(6 / fan_in)
            $limit = sqrt(6.0 / $in);
            $range = 2.0 * $limit;
            for ($i = 0; $i < $size; $i++) {
                $weights[] = -$limit + (mt_rand() / mt_getrandmax()) * $range;
            }
        } else {
            // Normal N(0, stddev), where stddev = sqrt(2 / fan_in)
            $stddev = sqrt(2.0 / $in);
            for ($i = 0; $i < $size; $i++) {
                $weights[] = XavierInitializer::randomNormal(0.0, $stddev);
            }
        }

        return $weights;
    }
}
