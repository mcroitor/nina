<?php

namespace mc\nina\schedule;

use mc\nina\exception\ValidationException;

/**
 * Exponential learning rate schedule.
 * 
 * Multiplies learning rate by gamma every epoch:
 * lr = baseLr * (gamma ^ epoch)
 * 
 * @package mc\nina\schedule
 * @author Mihail Croitor
 * @version 1.0.0
 */
class ExponentialSchedule implements LearningRateSchedule
{
    private float $gamma;

    /**
     * @param float $gamma Decay factor per epoch (0 < gamma < 1, default 0.95)
     */
    public function __construct(float $gamma = 0.95)
    {
        if ($gamma <= 0.0 || $gamma >= 1.0 || !is_finite($gamma)) {
            throw ValidationException::invalidValue('gamma', $gamma, 'float in (0, 1)');
        }
        $this->gamma = $gamma;
    }

    public function getLearningRate(int $epoch, float $baseLearningRate, ?float $currentLoss = null): float
    {
        return $baseLearningRate * pow($this->gamma, $epoch);
    }

    public function reset(): void
    {
    }
}
