<?php

namespace mc\nina\exception;

/**
 * Exception for network-level errors
 * 
 * Thrown when there are issues with neural network operations,
 * architecture, or configuration.
 */
class NetworkException extends NinaException
{
    protected string $severity = 'error';

    /**
     * Create exception for invalid network architecture
     */
    public static function invalidArchitecture(string $reason, array $context = []): self
    {
        return new self(
            "Invalid network architecture: {$reason}",
            1001,
            null,
            $context
        );
    }

    /**
     * Create exception for network configuration errors
     */
    public static function configurationError(string $parameter, $value, string $reason = ''): self
    {
        $message = "Network configuration error for parameter '{$parameter}' with value '{$value}'";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            1002,
            null,
            ['parameter' => $parameter, 'value' => $value]
        );
    }

    /**
     * Create exception for incompatible network operations
     */
    public static function incompatibleOperation(string $operation, string $reason): self
    {
        return new self(
            "Operation '{$operation}' is incompatible with current network state: {$reason}",
            1003,
            null,
            ['operation' => $operation]
        );
    }
}