<?php

namespace mc\nina\schedule;

/**
 * Interface for learning rate decay schedules.
 * 
 * Allows dynamic adjustment of learning rates during training epochs
 * to improve convergence and avoid overshooting local minima.
 * 
 * @package mc\nina\schedule
 * @author Mihail Croitor
 * @version 1.0.0
 */
interface LearningRateSchedule
{
    /**
     * Compute learning rate for given epoch and loss
     * 
     * @param int $epoch Current epoch number (1-indexed or 0-indexed)
     * @param float $baseLearningRate Initial/base learning rate
     * @param float|null $currentLoss Optional current epoch loss for adaptive schedules
     * 
     * @return float New effective learning rate
     */
    public function getLearningRate(int $epoch, float $baseLearningRate, ?float $currentLoss = null): float;

    /**
     * Reset schedule state (e.g., best loss, patience counters)
     */
    public function reset(): void;
}
