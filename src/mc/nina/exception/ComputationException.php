<?php

namespace mc\nina\exception;

/**
 * Exception for mathematical computation errors
 * 
 * Thrown when there are issues with mathematical operations,
 * matrix calculations, or numerical computation problems.
 */
class ComputationException extends NinaException
{
    protected string $severity = 'error';

    /**
     * Create exception for mathematical domain errors
     */
    public static function domainError(string $operation, $value, string $reason = ''): self
    {
        $message = "Mathematical domain error in {$operation} with value {$value}";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            5001,
            null,
            ['operation' => $operation, 'value' => $value]
        );
    }

    /**
     * Create exception for division by zero
     */
    public static function divisionByZero(string $context = ''): self
    {
        $message = "Division by zero error";
        if ($context) {
            $message .= " in {$context}";
        }
        
        return new self(
            $message,
            5002,
            null,
            ['context' => $context]
        );
    }

    /**
     * Create exception for matrix operation errors
     */
    public static function matrixError(string $operation, array $dimensions, string $reason = ''): self
    {
        $message = "Matrix operation '{$operation}' error with dimensions " . json_encode($dimensions);
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            5003,
            null,
            ['operation' => $operation, 'dimensions' => $dimensions]
        );
    }

    /**
     * Create exception for numerical overflow/underflow
     */
    public static function numericalOverflow(string $operation, $value, string $type = 'overflow'): self
    {
        return new self(
            "Numerical {$type} in {$operation} with value {$value}",
            5004,
            null,
            ['operation' => $operation, 'value' => $value, 'type' => $type]
        );
    }

    /**
     * Create exception for invalid numerical values (NaN, Infinity)
     */
    public static function invalidNumericalValue($value, string $context = ''): self
    {
        $valueType = is_infinite($value) ? 'infinity' : (is_nan($value) ? 'NaN' : 'invalid');
        $message = "Invalid numerical value ({$valueType})";
        if ($context) {
            $message .= " in {$context}";
        }
        
        return new self(
            $message,
            5005,
            null,
            ['value' => $value, 'value_type' => $valueType, 'context' => $context]
        );
    }
}