<?php

/**
 * Extended unit tests for Cell class backpropagation functionality
 * 
 * Tests the new backpropagation-related methods added to the Cell class:
 * - ForwardPass() method and caching
 * - Cached value getters (getLastWeightedSum, getLastActivation, getLastInputs)
 * - Activation derivative calculation
 * - Weight updates through gradients
 * - Cache management
 */

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
include_once __DIR__ . "/../src/mc/nina/exception/ComputationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/NetworkException.php";
include_once __DIR__ . "/../src/mc/nina/exception/SerializationException.php";
include_once __DIR__ . "/../src/mc/nina/activation/ActivationFunction.php";
include_once __DIR__ . "/../src/mc/nina/activation/ReLU.php";
include_once __DIR__ . "/../src/mc/nina/activation/Sigmoid.php";
include_once __DIR__ . "/../src/mc/nina/activation/Tanh.php";
include_once __DIR__ . "/../src/mc/nina/cell.php";

use \mc\nina\Cell;
use \mc\nina\exception\ValidationException;
use \mc\nina\exception\ComputationException;
use \mc\nina\exception\NetworkException;
use \mc\nina\activation\ReLU;
use \mc\nina\activation\Sigmoid;
use \mc\nina\activation\Tanh;

/**
 * Test ForwardPass method caching functionality
 */
function test_cell_forward_pass_caching(): bool {
    $result = true;

    $cell = new Cell("test_cell", [0.5, -0.3, 0.8], new Sigmoid());
    $input = [1.0, 0.5, -0.2];
    
    // Calculate expected values
    $expectedWeightedSum = 1.0 * 0.5 + 0.5 * (-0.3) + (-0.2) * 0.8; // = 0.19
    $expectedActivation = 1.0 / (1.0 + exp(-$expectedWeightedSum)); // Sigmoid activation
    
    // Perform forward pass
    $output = $cell->ForwardPass($input);
    
    // Test that output matches expected activation
    $result = $result && Assert::True(
        abs($output - $expectedActivation) < 1e-10,
        "ForwardPass output matches expected activation: {$output} ≈ {$expectedActivation}",
        "ForwardPass output mismatch: got {$output}, expected {$expectedActivation}"
    );
    
    // Test that cached values are correct
    $cachedWeightedSum = $cell->getLastWeightedSum();
    $cachedActivation = $cell->getLastActivation();
    $cachedInputs = $cell->getLastInputs();
    
    $result = $result && Assert::True(
        abs($cachedWeightedSum - $expectedWeightedSum) < 1e-10,
        "Cached weighted sum is correct: {$cachedWeightedSum} ≈ {$expectedWeightedSum}",
        "Cached weighted sum error: got {$cachedWeightedSum}, expected {$expectedWeightedSum}"
    );
    
    $result = $result && Assert::True(
        abs($cachedActivation - $expectedActivation) < 1e-10,
        "Cached activation is correct: {$cachedActivation} ≈ {$expectedActivation}",
        "Cached activation error: got {$cachedActivation}, expected {$expectedActivation}"
    );
    
    $result = $result && Assert::Equal($input, $cachedInputs);
    
    return $result;
}

/**
 * Test getActivationDerivative method
 */
function test_cell_activation_derivative(): bool {
    $result = true;

    // Test with Sigmoid activation
    $sigmoid_cell = new Cell("sigmoid_cell", [0.5], new Sigmoid());
    $input = [2.0];
    
    // Perform forward pass first
    $sigmoid_cell->ForwardPass($input);
    
    // Get derivative
    $derivative = $sigmoid_cell->getActivationDerivative();
    
    // Expected derivative for sigmoid at weighted sum = 1.0: sigmoid(1) * (1 - sigmoid(1))
    $weightedSum = 2.0 * 0.5; // = 1.0
    $sigmoid_value = 1.0 / (1.0 + exp(-$weightedSum));
    $expected_derivative = $sigmoid_value * (1.0 - $sigmoid_value);
    
    $result = $result && Assert::True(
        abs($derivative - $expected_derivative) < 1e-10,
        "Sigmoid derivative is correct: {$derivative} ≈ {$expected_derivative}",
        "Sigmoid derivative error: got {$derivative}, expected {$expected_derivative}"
    );
    
    // Test with ReLU activation
    $relu_cell = new Cell("relu_cell", [0.5], new ReLU());
    $relu_cell->ForwardPass([2.0]); // weighted sum = 1.0 > 0, so derivative = 1.0
    
    $relu_derivative = $relu_cell->getActivationDerivative();
    $result = $result && Assert::Equal(1.0, $relu_derivative);
    
    // Test ReLU with negative weighted sum
    $relu_cell->ForwardPass([-4.0]); // weighted sum = -2.0 < 0, so derivative = 0.0
    $relu_derivative_negative = $relu_cell->getActivationDerivative();
    $result = $result && Assert::Equal(0.0, $relu_derivative_negative);
    
    return $result;
}

/**
 * Test updateWeights method with gradient descent
 */
function test_cell_weight_updates(): bool {
    $result = true;

    $initial_weights = [0.5, -0.3, 0.8];
    $cell = new Cell("update_cell", $initial_weights);
    
    $gradients = [0.1, -0.05, 0.15];
    $learning_rate = 0.01;
    
    // Expected new weights: w_new = w_old - lr * gradient
    $expected_weights = [
        0.5 - 0.01 * 0.1,    // = 0.499
        -0.3 - 0.01 * (-0.05), // = -0.2995
        0.8 - 0.01 * 0.15     // = 0.7985
    ];
    
    // Apply weight updates
    $cell->updateWeights($gradients, $learning_rate);
    
    // Check that weights were updated correctly
    $updated_weights = $cell->GetDna();
    
    for ($i = 0; $i < count($expected_weights); $i++) {
        $result = $result && Assert::True(
            abs($updated_weights[$i] - $expected_weights[$i]) < 1e-10,
            "Weight[{$i}] updated correctly: {$updated_weights[$i]} ≈ {$expected_weights[$i]}",
            "Weight[{$i}] update error: got {$updated_weights[$i]}, expected {$expected_weights[$i]}"
        );
    }
    
    return $result;
}

/**
 * Test cache management and clearing
 */
function test_cell_cache_management(): bool {
    $result = true;

    $cell = new Cell("cache_cell", [0.5, -0.3], new Sigmoid());
    
    // Initially, cache should be empty - accessing cached values should throw exceptions
    try {
        $cell->getLastWeightedSum();
        $result = false; // Should not reach here
    } catch (NetworkException $e) {
        $result = $result && Assert::True(true, "Correctly throws exception when no forward pass performed");
    }
    
    try {
        $cell->getLastActivation();
        $result = false; // Should not reach here
    } catch (NetworkException $e) {
        $result = $result && Assert::True(true, "Correctly throws exception when accessing cached activation");
    }
    
    try {
        $cell->getLastInputs();
        $result = false; // Should not reach here
    } catch (NetworkException $e) {
        $result = $result && Assert::True(true, "Correctly throws exception when accessing cached inputs");
    }
    
    try {
        $cell->getActivationDerivative();
        $result = false; // Should not reach here
    } catch (NetworkException $e) {
        $result = $result && Assert::True(true, "Correctly throws exception when accessing derivative without forward pass");
    }
    
    // Perform forward pass to populate cache
    $input = [1.0, 0.5];
    $cell->ForwardPass($input);
    
    // Now cache should be populated
    $weightedSum = $cell->getLastWeightedSum();
    $activation = $cell->getLastActivation();
    $inputs = $cell->getLastInputs();
    $derivative = $cell->getActivationDerivative();
    
    $result = $result && Assert::True(is_numeric($weightedSum), "Cached weighted sum is numeric");
    $result = $result && Assert::True(is_numeric($activation), "Cached activation is numeric");
    $result = $result && Assert::Equal($input, $inputs);
    $result = $result && Assert::True(is_numeric($derivative), "Activation derivative is numeric");
    
    // Clear cache
    $cell->clearCache();
    
    // After clearing, cache should be empty again
    try {
        $cell->getLastWeightedSum();
        $result = false; // Should not reach here
    } catch (NetworkException $e) {
        $result = $result && Assert::True(true, "Cache cleared - weighted sum access throws exception");
    }
    
    return $result;
}

/**
 * Test error handling for updateWeights method
 */
function test_cell_update_weights_validation(): bool {
    $result = true;

    $cell = new Cell("validation_cell", [0.5, -0.3, 0.8]);
    
    // Test invalid learning rate
    try {
        $cell->updateWeights([0.1, 0.2, 0.3], 0.0); // learning rate = 0
        $result = false; // Should not reach here
    } catch (ComputationException $e) {
        $result = $result && Assert::True(true, "Correctly rejects zero learning rate");
    }
    
    try {
        $cell->updateWeights([0.1, 0.2, 0.3], -0.01); // negative learning rate
        $result = false; // Should not reach here
    } catch (ComputationException $e) {
        $result = $result && Assert::True(true, "Correctly rejects negative learning rate");
    }
    
    // Test mismatched gradient array size
    try {
        $cell->updateWeights([0.1, 0.2], 0.01); // 2 gradients for 3 weights
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::True(true, "Correctly rejects mismatched gradient array size");
    }
    
    // Test non-numeric gradients
    try {
        $cell->updateWeights([0.1, "invalid", 0.3], 0.01);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::True(true, "Correctly rejects non-numeric gradients");
    }
    
    // Test infinite gradients
    try {
        $cell->updateWeights([0.1, INF, 0.3], 0.01);
        $result = false; // Should not reach here
    } catch (ComputationException $e) {
        $result = $result && Assert::True(true, "Correctly rejects infinite gradients");
    }
    
    return $result;
}

/**
 * Test ForwardPass vs Calculate consistency
 */
function test_forward_pass_vs_calculate_consistency(): bool {
    $result = true;

    $cell = new Cell("consistency_cell", [0.2, 0.7, -0.4], new Tanh());
    $input = [1.0, -0.5, 0.8];
    
    // Both methods should produce the same output
    $forward_output = $cell->ForwardPass($input);
    $calculate_output = $cell->Calculate($input);
    
    $result = $result && Assert::True(
        abs($forward_output - $calculate_output) < 1e-10,
        "ForwardPass and Calculate produce same output: {$forward_output} ≈ {$calculate_output}",
        "ForwardPass/Calculate inconsistency: ForwardPass={$forward_output}, Calculate={$calculate_output}"
    );
    
    // Also test that CalculateRaw matches cached weighted sum
    $raw_output = $cell->CalculateRaw($input);
    $cached_weighted_sum = $cell->getLastWeightedSum();
    
    $result = $result && Assert::True(
        abs($raw_output - $cached_weighted_sum) < 1e-10,
        "CalculateRaw matches cached weighted sum: {$raw_output} ≈ {$cached_weighted_sum}",
        "CalculateRaw/cached inconsistency: raw={$raw_output}, cached={$cached_weighted_sum}"
    );
    
    return $result;
}

/**
 * Test complete backpropagation simulation
 */
function test_cell_backpropagation_simulation(): bool {
    $result = true;

    // Create a simple cell for testing complete backprop flow
    $cell = new Cell("backprop_cell", [0.5, -0.2], new Sigmoid());
    $input = [1.0, 0.5];
    
    // Step 1: Forward pass
    $output = $cell->ForwardPass($input);
    
    // Step 2: Simulate backward pass - calculate "error" gradient
    $target = 0.8;
    $error_gradient = 2 * ($output - $target); // Simple MSE derivative
    
    // Step 3: Calculate local gradients (chain rule)
    $activation_derivative = $cell->getActivationDerivative();
    $local_gradient = $error_gradient * $activation_derivative;
    
    // Step 4: Calculate weight gradients
    $cached_inputs = $cell->getLastInputs();
    $weight_gradients = [];
    foreach ($cached_inputs as $inp) {
        $weight_gradients[] = $local_gradient * $inp;
    }
    
    // Step 5: Update weights
    $learning_rate = 0.01;
    $old_weights = $cell->GetDna();
    $cell->updateWeights($weight_gradients, $learning_rate);
    $new_weights = $cell->GetDna();
    
    // Verify weights changed in the correct direction
    for ($i = 0; $i < count($old_weights); $i++) {
        $expected_change = -$learning_rate * $weight_gradients[$i];
        $actual_change = $new_weights[$i] - $old_weights[$i];
        
        $result = $result && Assert::True(
            abs($actual_change - $expected_change) < 1e-10,
            "Weight[{$i}] changed correctly: {$actual_change} ≈ {$expected_change}",
            "Weight[{$i}] change error: got {$actual_change}, expected {$expected_change}"
        );
    }
    
    // Test that a second forward pass shows different output (weights changed)
    $new_output = $cell->ForwardPass($input);
    $result = $result && Assert::True(
        $new_output != $output,
        "Output changed after weight update: {$new_output} != {$output}",
        "Output should change after weight update"
    );
    
    return $result;
}

// Main test execution
echo "=== Testing Cell Backpropagation Functionality ===\n\n";

$framework = new Framework();
$framework->Reset();

// Add tests to framework
$framework->AddTest("test_cell_forward_pass_caching");
$framework->AddTest("test_cell_activation_derivative");
$framework->AddTest("test_cell_weight_updates");
$framework->AddTest("test_cell_cache_management");
$framework->AddTest("test_cell_update_weights_validation");
$framework->AddTest("test_forward_pass_vs_calculate_consistency");
$framework->AddTest("test_cell_backpropagation_simulation");

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
    echo "\n🎉 All Cell backpropagation tests passed!\n";
    echo "✅ ForwardPass caching works correctly\n";
    echo "✅ Activation derivatives are computed properly\n";
    echo "✅ Weight updates via gradients function correctly\n";
    echo "✅ Cache management is robust\n";
    echo "✅ Input validation prevents errors\n";
    echo "✅ Consistency with existing methods maintained\n";
    echo "✅ Complete backpropagation flow verified\n";
} else {
    echo "\n❌ Some Cell backpropagation tests failed. Please check the implementation.\n";
}