<?php

namespace mc\nina\exception;

/**
 * Exception thrown for validation errors in the NiNa neural network library
 * 
 * This exception is used when input parameters fail validation checks,
 * such as invalid ranges, types, or constraints.
 */
class ValidationException extends NinaException
{
    protected string $severity = 'warning';

    /**
     * Create exception for invalid parameter value
     */
    public static function invalidValue(string $parameter, $value, string $expectedType = '', string $reason = ''): self
    {
        $message = "Invalid value for parameter '{$parameter}': {$value}";
        if ($expectedType) {
            $message .= " (expected {$expectedType})";
        }
        if ($reason) {
            $message .= " - {$reason}";
        }
        
        return new self(
            $message,
            6001,
            null,
            ['parameter' => $parameter, 'value' => $value, 'expected_type' => $expectedType]
        );
    }

    /**
     * Create exception for out of range values
     */
    public static function outOfRange(string $parameter, $value, $min = null, $max = null): self
    {
        $message = "Parameter '{$parameter}' value {$value} is out of range";
        if ($min !== null && $max !== null) {
            $message .= " (expected between {$min} and {$max})";
        } elseif ($min !== null) {
            $message .= " (expected >= {$min})";
        } elseif ($max !== null) {
            $message .= " (expected <= {$max})";
        }
        
        return new self(
            $message,
            6002,
            null,
            ['parameter' => $parameter, 'value' => $value, 'min' => $min, 'max' => $max]
        );
    }

    /**
     * Create exception for missing required parameters
     */
    public static function missingParameter(string $parameter, string $context = ''): self
    {
        $message = "Missing required parameter: {$parameter}";
        if ($context) {
            $message .= " in {$context}";
        }
        
        return new self(
            $message,
            6003,
            null,
            ['parameter' => $parameter, 'context' => $context]
        );
    }

    /**
     * Create exception for invalid array size
     */
    public static function invalidArraySize(string $parameter, int $actual, int $expected): self
    {
        return new self(
            "Invalid array size for '{$parameter}': expected {$expected}, got {$actual}",
            6004,
            null,
            ['parameter' => $parameter, 'actual' => $actual, 'expected' => $expected]
        );
    }
}