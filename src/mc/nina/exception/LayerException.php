<?php

namespace mc\nina\exception;

/**
 * Exception for layer-level errors
 * 
 * Thrown when there are issues with neural network layers,
 * their configuration, or operations.
 */
class LayerException extends NinaException
{
    protected string $severity = 'error';

    /**
     * Create exception for invalid layer size
     */
    public static function invalidSize(int $size, string $reason = ''): self
    {
        $message = "Invalid layer size: {$size}";
        if ($reason) {
            $message .= " - {$reason}";
        }
        
        return new self(
            $message,
            2001,
            null,
            ['size' => $size]
        );
    }

    /**
     * Create exception for layer dimension mismatch
     */
    public static function dimensionMismatch(int $expected, int $actual, string $context = ''): self
    {
        $message = "Layer dimension mismatch: expected {$expected}, got {$actual}";
        if ($context) {
            $message .= " in {$context}";
        }
        
        return new self(
            $message,
            2002,
            null,
            ['expected' => $expected, 'actual' => $actual, 'context' => $context]
        );
    }

    /**
     * Create exception for invalid layer connection
     */
    public static function invalidConnection(string $fromLayer, string $toLayer, string $reason): self
    {
        return new self(
            "Invalid layer connection from '{$fromLayer}' to '{$toLayer}': {$reason}",
            2003,
            null,
            ['from' => $fromLayer, 'to' => $toLayer]
        );
    }

    /**
     * Create exception for layer operation errors
     */
    public static function operationError(string $operation, string $reason): self
    {
        return new self(
            "Layer operation '{$operation}' failed: {$reason}",
            2004,
            null,
            ['operation' => $operation]
        );
    }
}