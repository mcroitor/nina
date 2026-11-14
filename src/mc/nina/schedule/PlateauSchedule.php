<?php

namespace mc\nina\schedule;

use mc\nina\exception\ValidationException;

/**
 * ReduceLROnPlateau schedule.
 * 
 * Reduces learning rate by factor when loss metric has stopped improving
 * for 'patience' consecutive epochs.
 * 
 * @package mc\nina\schedule
 * @author Mihail Croitor
 * @version 1.0.0
 */
class PlateauSchedule implements LearningRateSchedule
{
    private float $factor;
    private int $patience;
    private float $minDelta;
    private float $minLr;

    private ?float $bestLoss = null;
    private int $wait = 0;
    private ?float $currentLr = null;

    /**
     * @param float $factor Multiplicative decay factor (0 < factor < 1, default 0.5)
     * @param int $patience Number of epochs with no improvement before reducing (default 5)
     * @param float $minDelta Minimum change to qualify as an improvement (default 1e-4)
     * @param float $minLr Minimum allowed learning rate (default 1e-6)
     */
    public function __construct(
        float $factor = 0.5,
        int $patience = 5,
        float $minDelta = 1e-4,
        float $minLr = 1e-6
    ) {
        if ($factor <= 0.0 || $factor >= 1.0 || !is_finite($factor)) {
            throw ValidationException::invalidValue('factor', $factor, 'float in (0, 1)');
        }
        if ($patience <= 0) {
            throw ValidationException::invalidValue('patience', $patience, 'positive integer');
        }
        if ($minDelta < 0.0 || !is_finite($minDelta)) {
            throw ValidationException::invalidValue('minDelta', $minDelta, 'non-negative float');
        }
        if ($minLr <= 0.0 || !is_finite($minLr)) {
            throw ValidationException::invalidValue('minLr', $minLr, 'positive float');
        }

        $this->factor = $factor;
        $this->patience = $patience;
        $this->minDelta = $minDelta;
        $this->minLr = $minLr;
    }

    public function getLearningRate(int $epoch, float $baseLearningRate, ?float $currentLoss = null): float
    {
        if ($this->currentLr === null) {
            $this->currentLr = $baseLearningRate;
        }

        if ($currentLoss === null) {
            return $this->currentLr;
        }

        if ($this->bestLoss === null || ($this->bestLoss - $currentLoss) > $this->minDelta) {
            $this->bestLoss = $currentLoss;
            $this->wait = 0;
        } else {
            $this->wait++;
            if ($this->wait >= $this->patience) {
                $this->currentLr = max($this->minLr, $this->currentLr * $this->factor);
                $this->wait = 0;
            }
        }

        return $this->currentLr;
    }

    public function reset(): void
    {
        $this->bestLoss = null;
        $this->wait = 0;
        $this->currentLr = null;
    }
}
