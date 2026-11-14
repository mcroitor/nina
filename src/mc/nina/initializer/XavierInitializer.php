<?php

namespace mc\nina\initializer;

use mc\nina\exception\ValidationException;

/**
 * Xavier (Glorot) weight initializer.
 * 
 * Optimized for networks using Sigmoid or Tanh activation functions.
 * Maintains consistent variance of activations and gradients across layers.
 * 
 * Reference: Xavier Glorot & Yoshua Bengio, 2010 -
 * "Understanding the difficulty of training deep feedforward neural networks"
 * 
 * @package mc\nina\initializer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class XavierInitializer implements Initializer
{
    private bool $uniform;

    /**
     * @param bool $uniform True for Xavier Uniform, false for Xavier Normal (default true)
     */
    public function __construct(bool $uniform = true)
    {
        $this->uniform = $uniform;
    }

    public function getName(): string
    {
        return $this->uniform ? 'xavier_uniform' : 'xavier_normal';
    }

    public function initialize(int $size, int $fanIn = 0, int $fanOut = 0): array
    {
        if ($size <= 0) {
            throw ValidationException::invalidValue('size', $size, 'positive integer');
        }

        $in = ($fanIn > 0) ? $fanIn : $size;
        $out = ($fanOut > 0) ? $fanOut : $size;

        $weights = [];

        if ($this->uniform) {
            // Uniform in [-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))
            $limit = sqrt(6.0 / ($in + $out));
            $range = 2.0 * $limit;
            for ($i = 0; $i < $size; $i++) {
                $weights[] = -$limit + (mt_rand() / mt_getrandmax()) * $range;
            }
        } else {
            // Normal N(0, stddev), where stddev = sqrt(2 / (fan_in + fan_out))
            $stddev = sqrt(2.0 / ($in + $out));
            for ($i = 0; $i < $size; $i++) {
                $weights[] = self::randomNormal(0.0, $stddev);
            }
        }

        return $weights;
    }

    /**
     * Generates a normally distributed random number using Box-Muller transform
     */
    public static function randomNormal(float $mean = 0.0, float $stddev = 1.0): float
    {
        $u1 = max(1e-15, mt_rand() / mt_getrandmax());
        $u2 = mt_rand() / mt_getrandmax();
        $z0 = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
        return $mean + $z0 * $stddev;
    }
}
