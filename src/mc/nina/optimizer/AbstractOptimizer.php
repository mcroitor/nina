<?php

namespace mc\nina\optimizer;

use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Base abstract class for neural network optimizers.
 * 
 * Provides common functionality for regularization (L1 / L2 weight decay)
 * and parameter validation across optimizer implementations.
 * 
 * @package mc\nina\optimizer
 * @author Mihail Croitor
 * @version 1.0.0
 */
abstract class AbstractOptimizer implements Optimizer
{
    protected float $weightDecay = 0.0;
    protected float $l1 = 0.0;

    public function __construct(float $weightDecay = 0.0, float $l1 = 0.0)
    {
        $this->setWeightDecay($weightDecay);
        $this->setL1($l1);
    }

    public function setWeightDecay(float $weightDecay): void
    {
        if ($weightDecay < 0.0 || !is_finite($weightDecay)) {
            throw ValidationException::invalidValue('weightDecay', $weightDecay, 'non-negative finite float');
        }
        $this->weightDecay = $weightDecay;
    }

    public function getWeightDecay(): float
    {
        return $this->weightDecay;
    }

    public function setL1(float $l1): void
    {
        if ($l1 < 0.0 || !is_finite($l1)) {
            throw ValidationException::invalidValue('l1', $l1, 'non-negative finite float');
        }
        $this->l1 = $l1;
    }

    public function getL1(): float
    {
        return $this->l1;
    }

    /**
     * Applies L1 and L2 regularization to a gradient.
     * 
     * @param float $gradient Raw gradient
     * @param float $weight Current weight value
     * 
     * @return float Regularized gradient
     */
    protected function applyRegularization(float $gradient, float $weight): float
    {
        $reg = $gradient;
        if ($this->weightDecay > 0.0) {
            $reg += $this->weightDecay * $weight;
        }
        if ($this->l1 > 0.0) {
            $sign = ($weight > 0.0) ? 1.0 : (($weight < 0.0) ? -1.0 : 0.0);
            $reg += $this->l1 * $sign;
        }
        return $reg;
    }

    /**
     * Validates learning rate value.
     * 
     * @param float $learningRate
     * 
     * @throws ComputationException
     */
    protected function validateLearningRate(float $learningRate): void
    {
        if (!is_finite($learningRate) || $learningRate <= 0.0) {
            throw ComputationException::invalidNumericalValue(
                $learningRate,
                'learning rate (must be positive and finite)'
            );
        }
    }
}
