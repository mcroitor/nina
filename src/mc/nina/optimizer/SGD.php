<?php

namespace mc\nina\optimizer;

use mc\nina\Cell;
use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 * 
 * Supports standard SGD, classical momentum, Nesterov accelerated gradient,
 * and L1/L2 regularization (weight decay).
 * 
 * @package mc\nina\optimizer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class SGD extends AbstractOptimizer
{
    private float $momentum;
    private bool $nesterov;
    /** @var array<string, array<float>> Velocity buffer indexed by cell name */
    private array $velocities = [];

    /**
     * @param float $momentum Momentum factor (0 <= momentum < 1, default 0.0)
     * @param bool $nesterov Enables Nesterov momentum (default false)
     * @param float $weightDecay L2 regularization factor (default 0.0)
     * @param float $l1 L1 regularization factor (default 0.0)
     */
    public function __construct(
        float $momentum = 0.0,
        bool $nesterov = false,
        float $weightDecay = 0.0,
        float $l1 = 0.0
    ) {
        parent::__construct($weightDecay, $l1);
        $this->setMomentum($momentum);
        $this->nesterov = $nesterov;
    }

    public function getName(): string
    {
        return 'sgd';
    }

    public function setMomentum(float $momentum): void
    {
        if ($momentum < 0.0 || $momentum >= 1.0 || !is_finite($momentum)) {
            throw ValidationException::invalidValue('momentum', $momentum, 'float in [0, 1)');
        }
        $this->momentum = $momentum;
    }

    public function getMomentum(): float
    {
        return $this->momentum;
    }

    public function isNesterov(): bool
    {
        return $this->nesterov;
    }

    public function setNesterov(bool $nesterov): void
    {
        $this->nesterov = $nesterov;
    }

    public function update(Cell $cell, array $gradients, float $learningRate): void
    {
        $this->validateLearningRate($learningRate);
        
        $dna = $cell->GetDna();
        $size = count($dna);

        if (count($gradients) !== $size) {
            throw ValidationException::invalidArraySize('gradients', count($gradients), $size);
        }

        $cellKey = $cell->GetName();
        if ($this->momentum > 0.0 && !isset($this->velocities[$cellKey])) {
            $this->velocities[$cellKey] = array_fill(0, $size, 0.0);
        }

        $deltas = [];
        for ($i = 0; $i < $size; $i++) {
            if (!is_numeric($gradients[$i])) {
                throw ValidationException::invalidValue("gradients[{$i}]", $gradients[$i], 'numeric');
            }
            $g = (float)$gradients[$i];
            if (!is_finite($g)) {
                throw ComputationException::invalidNumericalValue($g, "gradient at index {$i}");
            }

            // Apply regularization (L1 / L2)
            $g = $this->applyRegularization($g, $dna[$i]);

            if ($this->momentum > 0.0) {
                // v = momentum * v + g
                $this->velocities[$cellKey][$i] = $this->momentum * $this->velocities[$cellKey][$i] + $g;
                if ($this->nesterov) {
                    $step = $this->momentum * $this->velocities[$cellKey][$i] + $g;
                } else {
                    $step = $this->velocities[$cellKey][$i];
                }
            } else {
                $step = $g;
            }

            // Weight delta: -learningRate * step
            $deltas[] = -$learningRate * $step;
        }

        $cell->applyDeltas($deltas);
    }

    public function reset(): void
    {
        $this->velocities = [];
    }
}
