<?php

namespace mc\nina\optimizer;

use mc\nina\Cell;
use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * RMSprop (Root Mean Square Propagation) optimizer.
 * 
 * Divides the gradient by a running average of its recent magnitude,
 * resolving the vanishing/exploding learning rate problem.
 * 
 * Reference: Geoff Hinton, Lecture 6e, Coursera Class Neural Networks for Machine Learning.
 * 
 * @package mc\nina\optimizer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class RMSprop extends AbstractOptimizer
{
    private float $alpha;
    private float $epsilon;
    private float $momentum;

    /** @var array<string, array<float>> Moving average of squared gradients */
    private array $v = [];
    /** @var array<string, array<float>> Momentum buffer */
    private array $momentumBuffer = [];

    /**
     * @param float $alpha Smoothing constant / decay factor (default 0.99)
     * @param float $epsilon Small constant for numerical stability (default 1e-8)
     * @param float $momentum Momentum factor (default 0.0)
     * @param float $weightDecay L2 regularization factor (default 0.0)
     * @param float $l1 L1 regularization factor (default 0.0)
     */
    public function __construct(
        float $alpha = 0.99,
        float $epsilon = 1e-8,
        float $momentum = 0.0,
        float $weightDecay = 0.0,
        float $l1 = 0.0
    ) {
        parent::__construct($weightDecay, $l1);
        $this->setAlpha($alpha);
        $this->setEpsilon($epsilon);
        $this->setMomentum($momentum);
    }

    public function getName(): string
    {
        return 'rmsprop';
    }

    public function setAlpha(float $alpha): void
    {
        if ($alpha <= 0.0 || $alpha >= 1.0 || !is_finite($alpha)) {
            throw ValidationException::invalidValue('alpha', $alpha, 'float in (0, 1)');
        }
        $this->alpha = $alpha;
    }

    public function getAlpha(): float
    {
        return $this->alpha;
    }

    public function setEpsilon(float $epsilon): void
    {
        if ($epsilon <= 0.0 || !is_finite($epsilon)) {
            throw ValidationException::invalidValue('epsilon', $epsilon, 'positive float');
        }
        $this->epsilon = $epsilon;
    }

    public function getEpsilon(): float
    {
        return $this->epsilon;
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

    public function update(Cell $cell, array $gradients, float $learningRate): void
    {
        $this->validateLearningRate($learningRate);

        $dna = $cell->GetDna();
        $size = count($dna);

        if (count($gradients) !== $size) {
            throw ValidationException::invalidArraySize('gradients', count($gradients), $size);
        }

        $cellKey = $cell->GetName();

        if (!isset($this->v[$cellKey])) {
            $this->v[$cellKey] = array_fill(0, $size, 0.0);
            if ($this->momentum > 0.0) {
                $this->momentumBuffer[$cellKey] = array_fill(0, $size, 0.0);
            }
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

            // Update moving average of squared gradients: v = alpha * v + (1 - alpha) * g^2
            $this->v[$cellKey][$i] = $this->alpha * $this->v[$cellKey][$i] + (1.0 - $this->alpha) * ($g * $g);

            $normG = $g / (sqrt($this->v[$cellKey][$i]) + $this->epsilon);

            if ($this->momentum > 0.0) {
                $this->momentumBuffer[$cellKey][$i] = $this->momentum * $this->momentumBuffer[$cellKey][$i] + $normG;
                $step = $this->momentumBuffer[$cellKey][$i];
            } else {
                $step = $normG;
            }

            $deltas[] = -$learningRate * $step;
        }

        $cell->applyDeltas($deltas);
    }

    public function reset(): void
    {
        $this->v = [];
        $this->momentumBuffer = [];
    }
}
