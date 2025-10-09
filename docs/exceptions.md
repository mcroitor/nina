# Custom Exceptions for NiNa Library

## Overview

The NiNa library uses a system of specialized exceptions to handle different types of errors. All custom exceptions inherit from the base `NinaException` class, which provides additional functionality for debugging and logging.

## Exception Hierarchy

```text
NinaException (abstract)
├── NetworkException
├── LayerException
├── TrainingException
├── SerializationException
├── ComputationException
└── ValidationException
```

## Base Class: NinaException

All custom exceptions inherit from this abstract base class.

### Features

- **Context Storage**: Stores additional information about error conditions
- **Severity Levels**: Categorizes errors by importance (error, warning, critical)
- **Detailed Messages**: Provides comprehensive error descriptions
- **Serialization**: Can be converted to array/JSON for logging
- **Unique Error Codes**: Each exception type has a specific code range

### Methods

```php
abstract class NinaException extends Exception
{
    public function getContext(): array;           // Additional error context
    public function getSeverity(): string;         // Error severity level
    public function getDetailedMessage(): string;  // Comprehensive error description
    public function toArray(): array;             // Serialize to array
}
```

## Exception Types

### ValidationException (6001-6099)
**Purpose**: Input validation and parameter checking errors

**Common Scenarios**:
- Invalid parameter values
- Out-of-range parameters
- Missing required parameters
- Array size mismatches

**Usage Example**:
```php
try {
    $cell = new Cell("", []); // Empty name and weights
} catch (ValidationException $e) {
    echo "Validation Error: " . $e->getMessage();
    echo "Error Code: " . $e->getCode();
    print_r($e->getContext());
}
```

### NetworkException (1001-1099)
**Purpose**: Neural network architecture and configuration errors

**Common Scenarios**:
- Invalid network architecture
- Incompatible layer connections
- Configuration parameter errors
- Network operation failures

**Usage Example**:
```php
try {
    $network = Network::Create("TestNet", 0, -5); // Invalid parameters
} catch (NetworkException $e) {
    echo "Network Error: " . $e->getDetailedMessage();
}
```

### LayerException (2001-2099)
**Purpose**: Layer-specific operation and configuration errors

**Common Scenarios**:
- Invalid layer sizes
- Dimension mismatches between layers
- Layer connection errors
- Layer operation failures

### TrainingException (3001-3099)
**Purpose**: Training process and genetic algorithm errors

**Common Scenarios**:
- Invalid training data
- Training convergence issues
- Population size errors
- Training parameter validation

### SerializationException (4001-4099)
**Purpose**: Model save/load and JSON operations errors

**Common Scenarios**:
- File operation errors
- JSON parsing/encoding failures
- Model version mismatches
- Data corruption during serialization

### ComputationException (5001-5099)
**Purpose**: Mathematical computation and numerical errors

**Common Scenarios**:
- Division by zero
- Numerical overflow/underflow
- Invalid mathematical operations
- NaN or infinite values

## Error Code Ranges

| Exception Type | Code Range | Purpose |
|----------------|------------|---------|
| NetworkException | 1001-1099 | Network architecture issues |
| LayerException | 2001-2099 | Layer configuration problems |
| TrainingException | 3001-3099 | Training process errors |
| SerializationException | 4001-4099 | Save/load operations |
| ComputationException | 5001-5099 | Mathematical computation errors |
| ValidationException | 6001-6099 | Input validation errors |

## Best Practices

### 1. Specific Exception Handling
```php
try {
    // Neural network operations
} catch (ValidationException $e) {
    // Handle input validation errors
    logger("Validation failed: " . $e->getMessage());
} catch (NetworkException $e) {
    // Handle network architecture errors
    logger("Network error: " . $e->getDetailedMessage());
} catch (NinaException $e) {
    // Handle any other library errors
    logger("General NiNa error: " . $e->getMessage());
}
```

### 2. Context Utilization
```php
try {
    $network->Calculate($invalidInput);
} catch (ComputationException $e) {
    $context = $e->getContext();
    echo "Operation: " . $context['operation'];
    echo "Invalid value: " . $context['value'];
    echo "Expected: " . $context['expected'];
}
```

### 3. Logging and Debugging
```php
try {
    // Library operations
} catch (NinaException $e) {
    // Log complete error information
    error_log(json_encode([
        'error' => $e->getMessage(),
        'code' => $e->getCode(),
        'severity' => $e->getSeverity(),
        'context' => $e->getContext(),
        'file' => $e->getFile(),
        'line' => $e->getLine()
    ]));
}
```

### 4. Recovery Strategies
```php
try {
    $trainer->Train($input, $expected);
} catch (TrainingException $e) {
    if ($e->getCode() === 3001) {
        // Invalid training data - try data preprocessing
        $cleanedInput = preprocessData($input);
        $trainer->Train($cleanedInput, $expected);
    } else {
        // Other training errors - abort training
        throw $e;
    }
}
```

## Integration with Library Components

### Cell Class Integration
- Validates weights array size and values
- Checks activation function compatibility
- Handles computation errors during calculation

### Layer Class Integration
- Validates cell array and layer configuration
- Handles dimension mismatches
- Manages layer operation errors

### Network Class Integration
- Validates network architecture
- Handles layer compatibility issues
- Manages network-wide operation errors

### Trainer Class Integration
- Validates training parameters and data
- Handles population management errors
- Manages genetic algorithm failures

### Predictor Class Integration
- Validates input data format
- Handles prediction computation errors
- Manages result formatting issues

## Error Recovery and Debugging

### Severity Levels
- **error**: Standard errors that can often be recovered from
- **warning**: Non-critical issues that don't stop execution
- **critical**: Serious errors that require immediate attention

### Context Information
Each exception provides relevant context:
- Parameter names and values that caused the error
- Expected vs. actual values
- Operation being performed when error occurred
- Additional debugging information

### Debugging Tips
1. **Check Error Codes**: Use specific error codes to identify exact issues
2. **Examine Context**: Review context data for parameter details
3. **Use Detailed Messages**: Get comprehensive error descriptions
4. **Log Everything**: Serialize exceptions to JSON for complete logging
5. **Handle Specifically**: Catch specific exception types for targeted recovery

---

*This exception system provides robust error handling and debugging capabilities for the NiNa neural network library.*