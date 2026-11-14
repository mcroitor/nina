<?php

namespace mc\nina\schedule;

use mc\nina\exception\ValidationException;

/**
 * Step Decay learning rate schedule.
 * 
 * Multiplies the learning rate by gamma every stepSize epochs:
 * lr = baseLr * (gamma ^ floor(epoch / stepSize))
 * 
 * @package mc\nina\schedule
 * @author Mihail Croitor
 * @version 1.0.0
 */
class StepDecaySchedule implements LearningRateSchedule
{
    private int $stepSize;
    private float $gamma;

    /**
     * @param int $stepSize Period of learning rate decay in epochs (must be > 0)
     * @param float $gamma Multiplicative factor of decay (0 < gamma < 1, default 0.5)
     */
    public function __construct(int $stepSize = 10, float $gamma = 0.5)
    {
        if ($stepSize <= 0) {
            throw ValidationException::invalidValue('stepSize', $stepSize, 'positive integer');
        }
        if ($gamma <= 0.0 || $gamma >= 1.0 || !is_finite($gamma)) {
            throw ValidationException::invalidValue('gamma', $gamma, 'float in (0, 1)');
        }
        $this->stepSize = $stepSize;
        $this->gamma = $gamma;
    }

    public function getLearningRate(int $epoch, float $baseLearningRate, ?float $currentLoss = null): float
    {
        $drops = (int)floor($epoch / $this->stepSize);
        return $baseLearningRate * pow($this->gamma, $drops);
    }

    public function reset(): void
    {
    }
}
