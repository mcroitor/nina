<?php

namespace mc\nina\exception;

/**
 * Exception for data serialization and deserialization errors
 * 
 * Thrown when there are issues saving or loading neural network models,
 * configuration files, or training data.
 */
class SerializationException extends NinaException
{
    protected string $severity = 'error';

    /**
     * Create exception for file operation errors
     */
    public static function fileError(string $operation, string $filename, string $reason = ''): self
    {
        $message = "File {$operation} error for '{$filename}'";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            4001,
            null,
            ['operation' => $operation, 'filename' => $filename]
        );
    }

    /**
     * Create exception for invalid data format
     */
    public static function invalidFormat(string $expectedFormat, string $actualFormat, string $context = ''): self
    {
        $message = "Invalid data format: expected {$expectedFormat}, got {$actualFormat}";
        if ($context) {
            $message .= " in {$context}";
        }
        
        return new self(
            $message,
            4002,
            null,
            ['expected' => $expectedFormat, 'actual' => $actualFormat, 'context' => $context]
        );
    }

    /**
     * Create exception for model version incompatibility
     */
    public static function versionMismatch(string $modelVersion, string $libraryVersion): self
    {
        return new self(
            "Model version incompatibility: model version {$modelVersion} is not compatible with library version {$libraryVersion}",
            4003,
            null,
            ['model_version' => $modelVersion, 'library_version' => $libraryVersion]
        );
    }

    /**
     * Create exception for corrupted data
     */
    public static function corruptedData(string $dataType, string $reason = ''): self
    {
        $message = "Corrupted {$dataType} data detected";
        if ($reason) {
            $message .= ": {$reason}";
        }
        
        return new self(
            $message,
            4004,
            null,
            ['data_type' => $dataType]
        );
    }

    /**
     * Create exception for JSON encoding/decoding errors
     */
    public static function jsonError(string $operation, int $jsonError): self
    {
        $errorMessages = [
            JSON_ERROR_NONE => 'No error',
            JSON_ERROR_DEPTH => 'Maximum stack depth exceeded',
            JSON_ERROR_STATE_MISMATCH => 'State mismatch (invalid or malformed JSON)',
            JSON_ERROR_CTRL_CHAR => 'Control character error',
            JSON_ERROR_SYNTAX => 'Syntax error',
            JSON_ERROR_UTF8 => 'Malformed UTF-8 characters'
        ];
        
        $errorMessage = $errorMessages[$jsonError] ?? "Unknown JSON error ({$jsonError})";
        
        return new self(
            "JSON {$operation} error: {$errorMessage}",
            4005,
            null,
            ['operation' => $operation, 'json_error' => $jsonError]
        );
    }
}