<?php

namespace mc\nina\schedule;

/**
 * Constant learning rate schedule.
 * 
 * Returns the base learning rate unchanged.
 * 
 * @package mc\nina\schedule
 * @author Mihail Croitor
 * @version 1.0.0
 */
class ConstantSchedule implements LearningRateSchedule
{
    public function getLearningRate(int $epoch, float $baseLearningRate, ?float $currentLoss = null): float
    {
        return $baseLearningRate;
    }

    public function reset(): void
    {
    }
}
