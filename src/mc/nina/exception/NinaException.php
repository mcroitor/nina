<?php

namespace mc\nina\exception;

/**
 * Base exception class for the NiNa neural network library
 * 
 * All custom exceptions in the library should extend this class
 * to provide consistent error handling and identification.
 */
abstract class NinaException extends \Exception
{
    /**
     * Error context data for debugging
     * @var array
     */
    protected array $context = [];

    /**
     * Error severity level
     * @var string
     */
    protected string $severity = 'error';

    public function __construct(string $message, int $code = 0, ?\Throwable $previous = null, array $context = [])
    {
        parent::__construct($message, $code, $previous);
        $this->context = $context;
    }

    /**
     * Get error context data
     * 
     * @return array Additional context information about the error
     */
    public function getContext(): array
    {
        return $this->context;
    }

    /**
     * Get error severity level
     * 
     * @return string The severity level (error, warning, critical)
     */
    public function getSeverity(): string
    {
        return $this->severity;
    }

    /**
     * Get a formatted error message with context
     * 
     * @return string Detailed error message including context
     */
    public function getDetailedMessage(): string
    {
        $message = $this->getMessage();
        
        if (!empty($this->context)) {
            $contextStr = json_encode($this->context, JSON_PRETTY_PRINT);
            $message .= "\nContext: " . $contextStr;
        }
        
        return $message;
    }

    /**
     * Convert exception to array for logging/serialization
     * 
     * @return array Exception data as array
     */
    public function toArray(): array
    {
        return [
            'type' => get_class($this),
            'message' => $this->getMessage(),
            'code' => $this->getCode(),
            'file' => $this->getFile(),
            'line' => $this->getLine(),
            'context' => $this->context,
            'severity' => $this->severity,
            'trace' => $this->getTraceAsString()
        ];
    }
}