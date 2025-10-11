<?php

/**
 * Unit tests for loss functions in backpropagation training
 * 
 * This test suite verifies the correctness of loss function calculations
 * and their derivatives for gradient computation.
 */

// include unit test framework
include_once __DIR__ . "/mc/unit/assert.php";
include_once __DIR__ . "/mc/unit/test.php";
include_once __DIR__ . "/mc/unit/framework.php";

use \mc\unit\Assert;
use \mc\unit\Test;
use \mc\unit\Framework;

// include tested classes
include_once __DIR__ . "/../src/mc/nina/exception/NinaException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ValidationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ComputationException.php";
include_once __DIR__ . "/../src/mc/nina/loss/LossFunction.php";
include_once __DIR__ . "/../src/mc/nina/loss/MeanSquaredError.php";
include_once __DIR__ . "/../src/mc/nina/loss/CrossEntropy.php";

use \mc\nina\loss\MeanSquaredError;
use \mc\nina\loss\CrossEntropy;
use \mc\nina\exception\ValidationException;

/**
 * Test MeanSquaredError loss calculation
 */
function test_mse_calculation(): bool {
    $result = true;
    $mse = new MeanSquaredError();
    
    // Test case 1: Perfect prediction (loss should be 0)
    $predicted = [1.0, 0.5, 0.8];
    $target = [1.0, 0.5, 0.8];
    $loss = $mse->calculate($predicted, $target);
    
    $result = $result && Assert::True(
        abs($loss) < 1e-10,
        "MSE with perfect prediction is zero: {$loss}",
        "MSE with perfect prediction should be zero, got: {$loss}"
    );
    
    // Test case 2: Known calculation
    $predicted = [0.8, 0.3, 0.9];
    $target = [1.0, 0.0, 1.0];
    $expected = (pow(0.8-1.0, 2) + pow(0.3-0.0, 2) + pow(0.9-1.0, 2)) / 3; // = (0.04 + 0.09 + 0.01) / 3 = 0.04667
    $loss = $mse->calculate($predicted, $target);
    
    $result = $result && Assert::True(
        abs($loss - $expected) < 1e-6,
        "MSE calculation matches expected value: {$loss} ≈ {$expected}",
        "MSE calculation error: got {$loss}, expected {$expected}"
    );
    
    return $result;
}

/**
 * Test MeanSquaredError derivative calculation
 */
function test_mse_derivative(): bool {
    $result = true;
    $mse = new MeanSquaredError();
    
    // Test case: derivatives should be 2/n * (predicted - target)
    $predicted = [0.8, 0.3];
    $target = [1.0, 0.0];
    $n = count($predicted);
    
    $gradients = $mse->derivative($predicted, $target);
    $expected = [
        (2.0 / $n) * (0.8 - 1.0), // = (2/2) * (-0.2) = -0.4
        (2.0 / $n) * (0.3 - 0.0)  // = (2/2) * (0.3) = 0.6
    ];
    
    $result = $result && Assert::Equal(count($expected), count($gradients));
    
    for ($i = 0; $i < count($expected); $i++) {
        $result = $result && Assert::True(
            abs($gradients[$i] - $expected[$i]) < 1e-10,
            "MSE gradient[{$i}] = {$gradients[$i]} ≈ {$expected[$i]}",
            "MSE gradient[{$i}] error: got {$gradients[$i]}, expected {$expected[$i]}"
        );
    }
    
    return $result;
}

/**
 * Test CrossEntropy loss calculation
 */
function test_cross_entropy_calculation(): bool {
    $result = true;
    $ce = new CrossEntropy();
    
    // Test case 1: Perfect prediction (loss should be 0)
    $predicted = [1.0, 0.0, 0.0];
    $target = [1.0, 0.0, 0.0];
    $loss = $ce->calculate($predicted, $target);
    
    $result = $result && Assert::True(
        abs($loss) < 1e-10,
        "Cross-entropy with perfect prediction is near zero: {$loss}",
        "Cross-entropy with perfect prediction should be near zero, got: {$loss}"
    );
    
    // Test case 2: Known calculation - single class
    $predicted = [0.7];
    $target = [1.0];
    $expected = -log(0.7); // ≈ 0.357
    $loss = $ce->calculate($predicted, $target);
    
    $result = $result && Assert::True(
        abs($loss - $expected) < 1e-6,
        "Cross-entropy calculation matches expected: {$loss} ≈ {$expected}",
        "Cross-entropy calculation error: got {$loss}, expected {$expected}"
    );
    
    return $result;
}

/**
 * Test CrossEntropy derivative calculation
 */
function test_cross_entropy_derivative(): bool {
    $result = true;
    $ce = new CrossEntropy();
    
    // Test case: derivatives should be -target[i] / predicted[i]
    $predicted = [0.8, 0.2];
    $target = [1.0, 0.0];
    
    $gradients = $ce->derivative($predicted, $target);
    $expected = [
        -1.0 / 0.8,  // = -1.25
        -0.0 / 0.2   // = 0.0
    ];
    
    $result = $result && Assert::Equal(count($expected), count($gradients));
    
    for ($i = 0; $i < count($expected); $i++) {
        $result = $result && Assert::True(
            abs($gradients[$i] - $expected[$i]) < 1e-10,
            "Cross-entropy gradient[{$i}] = {$gradients[$i]} ≈ {$expected[$i]}",
            "Cross-entropy gradient[{$i}] error: got {$gradients[$i]}, expected {$expected[$i]}"
        );
    }
    
    return $result;
}

/**
 * Test loss function validation behavior
 */
function test_loss_function_validation(): bool {
    $result = true;
    $mse = new MeanSquaredError();
    $ce = new CrossEntropy();
    
    // Test empty arrays
    try {
        $mse->calculate([], [1.0]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::True(true, "MSE correctly rejects empty predicted array");
    }
    
    // Test mismatched array sizes
    try {
        $mse->calculate([1.0, 2.0], [1.0]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::True(true, "MSE correctly rejects mismatched array sizes");
    }
    
    // Test invalid probability values for cross-entropy
    try {
        $ce->calculate([1.5], [1.0]); // predicted > 1.0
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::True(true, "Cross-entropy correctly rejects invalid probability values");
    }
    
    return $result;
}

/**
 * Test loss function properties and metadata
 */
function test_loss_function_properties(): bool {
    $result = true;
    
    $functions = [
        new MeanSquaredError(),
        new CrossEntropy()
    ];
    
    foreach ($functions as $func) {
        // Test getName() returns non-empty string
        $name = $func->getName();
        $result = $result && Assert::True(
            !empty($name) && is_string($name),
            "Loss function {$name} has valid name",
            "Loss function name is invalid"
        );
        
        // Test getInfo() returns proper array structure
        $info = $func->getInfo();
        $result = $result && Assert::True(
            is_array($info) && isset($info['name']),
            "Loss function {$name} has valid info structure",
            "Loss function {$name} info structure is invalid"
        );
        
        $result = $result && Assert::True(
            isset($info['range']) && is_array($info['range']),
            "Loss function {$name} has valid range in info",
            "Loss function {$name} range in info is invalid"
        );
    }
    
    return $result;
}

/**
 * Test categorical cross-entropy method
 */
function test_categorical_cross_entropy(): bool {
    $result = true;
    $ce = new CrossEntropy();
    
    // Test categorical cross-entropy for one-hot encoded case
    $predicted = [0.1, 0.8, 0.1]; // 3 classes
    $true_class = 1; // Second class is correct
    
    $loss = $ce->categoricalCrossEntropy($predicted, $true_class);
    $expected = -log(0.8);
    
    $result = $result && Assert::True(
        abs($loss - $expected) < 1e-10,
        "Categorical cross-entropy calculation: {$loss} ≈ {$expected}",
        "Categorical cross-entropy error: got {$loss}, expected {$expected}"
    );
    
    // Test edge case - very confident wrong prediction
    $predicted = [0.9, 0.05, 0.05];
    $true_class = 2; // Third class is correct, but prediction is confident about first
    
    $loss = $ce->categoricalCrossEntropy($predicted, $true_class);
    $expected = -log(0.05);
    
    $result = $result && Assert::True(
        abs($loss - $expected) < 1e-10,
        "Categorical cross-entropy with confident wrong prediction: {$loss} ≈ {$expected}",
        "Categorical cross-entropy confident wrong error: got {$loss}, expected {$expected}"
    );
    
    return $result;
}

// Main test execution
echo "=== Testing Loss Functions for Backpropagation ===\n\n";

$framework = new Framework();
$framework->Reset();

// Add tests to framework
$framework->AddTest("test_mse_calculation");
$framework->AddTest("test_mse_derivative");
$framework->AddTest("test_cross_entropy_calculation");
$framework->AddTest("test_cross_entropy_derivative");
$framework->AddTest("test_loss_function_validation");
$framework->AddTest("test_loss_function_properties");
$framework->AddTest("test_categorical_cross_entropy");

// Run all tests
$framework->Run();

// Print summary
echo "\n=== Test Results Summary ===\n";
echo "Total tests: " . $framework->TotalTests() . "\n";
echo "Passed: " . $framework->TotalPassed() . "\n";
echo "Failed: " . $framework->TotalFailed() . "\n";
echo "Total assertions: " . Assert::Total() . "\n";
echo "Passed assertions: " . Assert::Passed() . "\n";
echo "Failed assertions: " . Assert::Failed() . "\n";

if ($framework->TotalFailed() == 0 && Assert::Failed() == 0) {
    echo "\n🎉 All tests passed! Loss functions are correctly implemented.\n";
} else {
    echo "\n❌ Some tests failed. Please check the implementation.\n";
}