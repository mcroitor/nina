<?php

namespace mc\nina\exception;

/**
 * Exception for training-related errors
 * 
 * Thrown when there are issues during neural network training,
 * including data problems, convergence issues, or training configuration errors.
 */
class TrainingException extends NinaException
{
    protected string $severity = 'error';

    /**
     * Create exception for invalid training data
     */
    public static function invalidTrainingData(string $reason, array $context = []): self
    {
        return new self(
            "Invalid training data: {$reason}",
            3001,
            null,
            $context
        );
    }

    /**
     * Create exception for training convergence issues
     */
    public static function convergenceError(int $epoch, float $error, string $reason = ''): self
    {
        $message = "Training convergence error at epoch {$epoch} with error {$error}";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            3002,
            null,
            ['epoch' => $epoch, 'error' => $error]
        );
    }

    /**
     * Create exception for invalid training parameters
     */
    public static function invalidParameters(string $parameter, $value, string $reason = ''): self
    {
        $message = "Invalid training parameter '{$parameter}' with value '{$value}'";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            3003,
            null,
            ['parameter' => $parameter, 'value' => $value]
        );
    }

    /**
     * Create exception for training data size mismatch
     */
    public static function dataSizeMismatch(int $inputSize, int $expectedSize, string $dataType = 'input'): self
    {
        return new self(
            "Training data size mismatch for {$dataType}: expected {$expectedSize}, got {$inputSize}",
            3004,
            null,
            ['input_size' => $inputSize, 'expected_size' => $expectedSize, 'data_type' => $dataType]
        );
    }

    /**
     * Create exception for training timeout
     */
    public static function trainingTimeout(int $maxEpochs, float $timeElapsed): self
    {
        return new self(
            "Training timeout: reached maximum epochs ({$maxEpochs}) or time limit ({$timeElapsed}s)",
            3005,
            null,
            ['max_epochs' => $maxEpochs, 'time_elapsed' => $timeElapsed]
        );
    }
}