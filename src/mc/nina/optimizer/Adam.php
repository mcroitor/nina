<?php

namespace mc\nina\optimizer;

use mc\nina\Cell;
use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 * 
 * Computes individual adaptive learning rates for different parameters
 * from estimates of first and second moments of the gradients.
 * 
 * Reference: Kingma & Ba, 2014 - "Adam: A Method for Stochastic Optimization"
 * 
 * @package mc\nina\optimizer
 * @author Mihail Croitor
 * @version 1.0.0
 */
class Adam extends AbstractOptimizer
{
    private float $beta1;
    private float $beta2;
    private float $epsilon;

    /** @var array<string, array<float>> First moment estimates */
    private array $m = [];
    /** @var array<string, array<float>> Second moment estimates */
    private array $v = [];
    /** @var array<string, int> Time step counters */
    private array $t = [];

    /**
     * @param float $beta1 Exponential decay rate for first moment (default 0.9)
     * @param float $beta2 Exponential decay rate for second moment (default 0.999)
     * @param float $epsilon Small constant for numerical stability (default 1e-8)
     * @param float $weightDecay L2 regularization factor (default 0.0)
     * @param float $l1 L1 regularization factor (default 0.0)
     */
    public function __construct(
        float $beta1 = 0.9,
        float $beta2 = 0.999,
        float $epsilon = 1e-8,
        float $weightDecay = 0.0,
        float $l1 = 0.0
    ) {
        parent::__construct($weightDecay, $l1);
        $this->setBeta1($beta1);
        $this->setBeta2($beta2);
        $this->setEpsilon($epsilon);
    }

    public function getName(): string
    {
        return 'adam';
    }

    public function setBeta1(float $beta1): void
    {
        if ($beta1 <= 0.0 || $beta1 >= 1.0 || !is_finite($beta1)) {
            throw ValidationException::invalidValue('beta1', $beta1, 'float in (0, 1)');
        }
        $this->beta1 = $beta1;
    }

    public function getBeta1(): float
    {
        return $this->beta1;
    }

    public function setBeta2(float $beta2): void
    {
        if ($beta2 <= 0.0 || $beta2 >= 1.0 || !is_finite($beta2)) {
            throw ValidationException::invalidValue('beta2', $beta2, 'float in (0, 1)');
        }
        $this->beta2 = $beta2;
    }

    public function getBeta2(): float
    {
        return $this->beta2;
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

    public function update(Cell $cell, array $gradients, float $learningRate): void
    {
        $this->validateLearningRate($learningRate);

        $dna = $cell->GetDna();
        $size = count($dna);

        if (count($gradients) !== $size) {
            throw ValidationException::invalidArraySize('gradients', count($gradients), $size);
        }

        $cellKey = $cell->GetName();

        // Initialize state for new cell
        if (!isset($this->t[$cellKey])) {
            $this->t[$cellKey] = 0;
            $this->m[$cellKey] = array_fill(0, $size, 0.0);
            $this->v[$cellKey] = array_fill(0, $size, 0.0);
        }

        $this->t[$cellKey]++;
        $t = $this->t[$cellKey];

        // Bias correction factors
        $biasCorrection1 = 1.0 - pow($this->beta1, $t);
        $biasCorrection2 = 1.0 - pow($this->beta2, $t);

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

            // Update biased first moment estimate
            $this->m[$cellKey][$i] = $this->beta1 * $this->m[$cellKey][$i] + (1.0 - $this->beta1) * $g;

            // Update biased second raw moment estimate
            $this->v[$cellKey][$i] = $this->beta2 * $this->v[$cellKey][$i] + (1.0 - $this->beta2) * ($g * $g);

            // Compute bias-corrected estimates
            $mHat = $this->m[$cellKey][$i] / $biasCorrection1;
            $vHat = $this->v[$cellKey][$i] / $biasCorrection2;

            // Compute step: -learningRate * mHat / (sqrt(vHat) + epsilon)
            $step = $learningRate * $mHat / (sqrt($vHat) + $this->epsilon);
            $deltas[] = -$step;
        }

        $cell->applyDeltas($deltas);
    }

    public function reset(): void
    {
        $this->m = [];
        $this->v = [];
        $this->t = [];
    }
}
