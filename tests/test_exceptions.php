<?php

// include unit test framework
include_once __DIR__ . "/mc/unit/assert.php";
include_once __DIR__ . "/mc/unit/test.php";
include_once __DIR__ . "/mc/unit/framework.php";

use \mc\unit\Assert;
use \mc\unit\Test;
use \mc\unit\Framework;

// include tested classes
include_once __DIR__ . "/../src/mc/lib.php";
include_once __DIR__ . "/../src/mc/nina/exception/NinaException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ValidationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/NetworkException.php";
include_once __DIR__ . "/../src/mc/nina/exception/LayerException.php";
include_once __DIR__ . "/../src/mc/nina/exception/TrainingException.php";
include_once __DIR__ . "/../src/mc/nina/exception/SerializationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ComputationException.php";

use \mc\nina\exception\NinaException;
use \mc\nina\exception\ValidationException;
use \mc\nina\exception\NetworkException;
use \mc\nina\exception\LayerException;
use \mc\nina\exception\TrainingException;
use \mc\nina\exception\SerializationException;
use \mc\nina\exception\ComputationException;

function test_validation_exception_invalid_value(): bool {
    $result = true;
    
    $exception = ValidationException::invalidValue('test_param', 'invalid', 'string', 'wrong type');
    
    $result &= Assert::Equal($exception->getCode(), 6001);
    $result &= Assert::True(str_contains($exception->getMessage(), 'test_param'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'invalid'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'string'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'wrong type'));
    $result &= Assert::Equal($exception->getSeverity(), 'warning');
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['parameter'], 'test_param');
    $result &= Assert::Equal($context['value'], 'invalid');
    $result &= Assert::Equal($context['expected_type'], 'string');
    
    return $result;
}

function test_validation_exception_out_of_range(): bool {
    $result = true;
    
    $exception = ValidationException::outOfRange('size', 15, 1, 10);
    
    $result &= Assert::Equal($exception->getCode(), 6002);
    $result &= Assert::True(str_contains($exception->getMessage(), 'size'));
    $result &= Assert::True(str_contains($exception->getMessage(), '15'));
    $result &= Assert::True(str_contains($exception->getMessage(), '1'));
    $result &= Assert::True(str_contains($exception->getMessage(), '10'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['parameter'], 'size');
    $result &= Assert::Equal($context['value'], 15);
    $result &= Assert::Equal($context['min'], 1);
    $result &= Assert::Equal($context['max'], 10);
    
    return $result;
}

function test_network_exception_invalid_architecture(): bool {
    $result = true;
    
    $exception = NetworkException::invalidArchitecture('layers mismatch', ['layer_count' => 0]);
    
    $result &= Assert::Equal($exception->getCode(), 1001);
    $result &= Assert::True(str_contains($exception->getMessage(), 'layers mismatch'));
    $result &= Assert::Equal($exception->getSeverity(), 'error');
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['layer_count'], 0);
    
    return $result;
}

function test_network_exception_configuration_error(): bool {
    $result = true;
    
    $exception = NetworkException::configurationError('learning_rate', 0.5, 'must be positive');
    
    $result &= Assert::Equal($exception->getCode(), 1002);
    $result &= Assert::True(str_contains($exception->getMessage(), 'learning_rate'));
    $result &= Assert::True(str_contains($exception->getMessage(), '0.5'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'must be positive'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['parameter'], 'learning_rate');
    $result &= Assert::Equal($context['value'], 0.5);
    
    return $result;
}

function test_layer_exception_invalid_size(): bool {
    $result = true;
    
    $exception = LayerException::invalidSize(-5, 'size must be positive');
    
    $result &= Assert::Equal($exception->getCode(), 2001);
    $result &= Assert::True(str_contains($exception->getMessage(), '-5'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'size must be positive'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['size'], -5);
    
    return $result;
}

function test_layer_exception_dimension_mismatch(): bool {
    $result = true;
    
    $exception = LayerException::dimensionMismatch(10, 5, 'layer connection');
    
    $result &= Assert::Equal($exception->getCode(), 2002);
    $result &= Assert::True(str_contains($exception->getMessage(), '10'));
    $result &= Assert::True(str_contains($exception->getMessage(), '5'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'layer connection'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['expected'], 10);
    $result &= Assert::Equal($context['actual'], 5);
    $result &= Assert::Equal($context['context'], 'layer connection');
    
    return $result;
}

function test_training_exception_invalid_training_data(): bool {
    $result = true;
    
    $exception = TrainingException::invalidTrainingData('empty dataset', ['size' => 0]);
    
    $result &= Assert::Equal($exception->getCode(), 3001);
    $result &= Assert::True(str_contains($exception->getMessage(), 'empty dataset'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['size'], 0);
    
    return $result;
}

function test_training_exception_convergence_error(): bool {
    $result = true;
    
    $exception = TrainingException::convergenceError(100, 0.95, 'no improvement');
    
    $result &= Assert::Equal($exception->getCode(), 3002);
    $result &= Assert::True(str_contains($exception->getMessage(), '100'));
    $result &= Assert::True(str_contains($exception->getMessage(), '0.95'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'no improvement'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['epoch'], 100);
    $result &= Assert::Equal($context['error'], 0.95);
    
    return $result;
}

function test_serialization_exception_file_error(): bool {
    $result = true;
    
    $exception = SerializationException::fileError('read', 'model.json', 'file not found');
    
    $result &= Assert::Equal($exception->getCode(), 4001);
    $result &= Assert::True(str_contains($exception->getMessage(), 'read'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'model.json'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'file not found'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['operation'], 'read');
    $result &= Assert::Equal($context['filename'], 'model.json');
    
    return $result;
}

function test_serialization_exception_invalid_format(): bool {
    $result = true;
    
    $exception = SerializationException::invalidFormat('JSON', 'XML', 'model file');
    
    $result &= Assert::Equal($exception->getCode(), 4002);
    $result &= Assert::True(str_contains($exception->getMessage(), 'JSON'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'XML'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'model file'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['expected'], 'JSON');
    $result &= Assert::Equal($context['actual'], 'XML');
    $result &= Assert::Equal($context['context'], 'model file');
    
    return $result;
}

function test_computation_exception_domain_error(): bool {
    $result = true;
    
    $exception = ComputationException::domainError('sqrt', -5, 'negative value');
    
    $result &= Assert::Equal($exception->getCode(), 5001);
    $result &= Assert::True(str_contains($exception->getMessage(), 'sqrt'));
    $result &= Assert::True(str_contains($exception->getMessage(), '-5'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'negative value'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['operation'], 'sqrt');
    $result &= Assert::Equal($context['value'], -5);
    
    return $result;
}

function test_computation_exception_division_by_zero(): bool {
    $result = true;
    
    $exception = ComputationException::divisionByZero('weight calculation');
    
    $result &= Assert::Equal($exception->getCode(), 5002);
    $result &= Assert::True(str_contains($exception->getMessage(), 'Division by zero'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'weight calculation'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['context'], 'weight calculation');
    
    return $result;
}

function test_nina_exception_base_features(): bool {
    $result = true;
    
    // Create a concrete exception for testing base features
    $exception = ValidationException::invalidValue('test', 'value', 'type');
    
    // Test base NinaException methods
    $result &= Assert::True($exception instanceof NinaException);
    $result &= Assert::True(is_array($exception->getContext()));
    $result &= Assert::True(is_string($exception->getSeverity()));
    $result &= Assert::True(is_string($exception->getDetailedMessage()));
    $result &= Assert::True(is_array($exception->toArray()));
    
    // Test detailed message includes context
    $detailedMessage = $exception->getDetailedMessage();
    $result &= Assert::True(str_contains($detailedMessage, $exception->getMessage()));
    
    // Test toArray structure
    $array = $exception->toArray();
    $result &= Assert::True(isset($array['type']));
    $result &= Assert::True(isset($array['message']));
    $result &= Assert::True(isset($array['code']));
    $result &= Assert::True(isset($array['file']));
    $result &= Assert::True(isset($array['line']));
    $result &= Assert::True(isset($array['context']));
    $result &= Assert::True(isset($array['severity']));
    $result &= Assert::True(isset($array['trace']));
    
    return $result;
}

function test_validation_exception_missing_parameter(): bool {
    $result = true;
    
    $exception = ValidationException::missingParameter('required_param', 'constructor');
    
    $result &= Assert::Equal($exception->getCode(), 6003);
    $result &= Assert::True(str_contains($exception->getMessage(), 'required_param'));
    $result &= Assert::True(str_contains($exception->getMessage(), 'constructor'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['parameter'], 'required_param');
    $result &= Assert::Equal($context['context'], 'constructor');
    
    return $result;
}

function test_validation_exception_invalid_array_size(): bool {
    $result = true;
    
    $exception = ValidationException::invalidArraySize('data', 5, 10);
    
    $result &= Assert::Equal($exception->getCode(), 6004);
    $result &= Assert::True(str_contains($exception->getMessage(), 'data'));
    $result &= Assert::True(str_contains($exception->getMessage(), '5'));
    $result &= Assert::True(str_contains($exception->getMessage(), '10'));
    
    $context = $exception->getContext();
    $result &= Assert::Equal($context['parameter'], 'data');
    $result &= Assert::Equal($context['actual'], 5);
    $result &= Assert::Equal($context['expected'], 10);
    
    return $result;
}

// Run the tests
echo "Testing NiNa Exception System..." . PHP_EOL;

$framework = new Framework();

$framework->AddTest("test_validation_exception_invalid_value");
$framework->AddTest("test_validation_exception_out_of_range");
$framework->AddTest("test_validation_exception_missing_parameter");
$framework->AddTest("test_validation_exception_invalid_array_size");

$framework->AddTest("test_network_exception_invalid_architecture");
$framework->AddTest("test_network_exception_configuration_error");

$framework->AddTest("test_layer_exception_invalid_size");
$framework->AddTest("test_layer_exception_dimension_mismatch");

$framework->AddTest("test_training_exception_invalid_training_data");
$framework->AddTest("test_training_exception_convergence_error");

$framework->AddTest("test_serialization_exception_file_error");
$framework->AddTest("test_serialization_exception_invalid_format");

$framework->AddTest("test_computation_exception_domain_error");
$framework->AddTest("test_computation_exception_division_by_zero");

$framework->AddTest("test_nina_exception_base_features");

$framework->Run();
$framework->PrintInfo();

echo "Exception tests completed!" . PHP_EOL;