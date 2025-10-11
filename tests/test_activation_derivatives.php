<?php

/**
 * Unit tests for activation function derivatives in backpropagation context
 * 
 * This test suite verifies the mathematical correctness of activation function
 * derivatives using numerical differentiation for comparison.
 */

// include unit test framework
include_once __DIR__ . "/mc/unit/assert.php";
include_once __DIR__ . "/mc/unit/test.php";
include_once __DIR__ . "/mc/unit/framework.php";

use \mc\unit\Assert;
use \mc\unit\Test;
use \mc\unit\Framework;

// include tested classes
include_once __DIR__ . "/../src/mc/nina/activation/ActivationFunction.php";
include_once __DIR__ . "/../src/mc/nina/activation/Sigmoid.php";
include_once __DIR__ . "/../src/mc/nina/activation/Tanh.php";
include_once __DIR__ . "/../src/mc/nina/activation/ReLU.php";
include_once __DIR__ . "/../src/mc/nina/activation/Softmax.php";

use \mc\nina\activation\Sigmoid;
use \mc\nina\activation\Tanh;
use \mc\nina\activation\ReLU;
use \mc\nina\activation\Softmax;

/**
 * Numerical derivative calculation for comparison with analytical derivatives
 * 
 * @param callable $function The function to differentiate
 * @param float $x The point at which to calculate derivative
 * @param float $h Step size for numerical differentiation
 * @return float Numerical derivative approximation
 */
function numericalDerivative(callable $function, float $x, float $h = 1e-7): float {
    return ($function($x + $h) - ($function($x - $h))) / (2 * $h);
}

/**
 * Test Sigmoid activation function derivative accuracy
 */
function test_sigmoid_derivative(): bool {
    $result = true;
    $sigmoid = new Sigmoid();
    
    // Test values covering different ranges
    $testValues = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    
    foreach ($testValues as $x) {
        $analytical = $sigmoid->derivative($x);
        $numerical = numericalDerivative(
            function($val) use ($sigmoid) { return $sigmoid->activate($val); }, 
            $x
        );
        
        $error = abs($analytical - $numerical);
        $isAccurate = $error < 1e-5;
        
        $result = $result && Assert::True(
            $isAccurate,
            "Sigmoid derivative at x={$x}: analytical={$analytical}, numerical={$numerical}, error={$error}",
            "Sigmoid derivative FAILED at x={$x}: error={$error} > 1e-5"
        );
    }
    
    return $result;
}

/**
 * Test Tanh activation function derivative accuracy
 */
function test_tanh_derivative(): bool {
    $result = true;
    $tanh = new Tanh();
    
    // Test values covering different ranges
    $testValues = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    
    foreach ($testValues as $x) {
        $analytical = $tanh->derivative($x);
        $numerical = numericalDerivative(
            fn($val) => $tanh->activate($val), 
            $x
        );
        
        $error = abs($analytical - $numerical);
        $isAccurate = $error < 1e-5;
        
        $result = $result && Assert::True(
            $isAccurate,
            "Tanh derivative at x={$x}: analytical={$analytical}, numerical={$numerical}, error={$error}",
            "Tanh derivative FAILED at x={$x}: error={$error} > 1e-5"
        );
    }
    
    return $result;
}

/**
 * Test ReLU activation function derivative accuracy
 */
function test_relu_derivative(): bool {
    $result = true;
    $relu = new ReLU();
    
    // Test values, including edge case around 0
    $testValues = [-1.0, -0.1, 0.1, 1.0, 2.0];
    
    foreach ($testValues as $x) {
        $analytical = $relu->derivative($x);
        $numerical = numericalDerivative(
            fn($val) => $relu->activate($val), 
            $x
        );
        
        $error = abs($analytical - $numerical);
        // ReLU derivative is exact except at x=0 where it's undefined
        // Use looser tolerance for numerical precision issues
        $tolerance = 1e-9;
        $isAccurate = $error < $tolerance;
        
        $result = $result && Assert::True(
            $isAccurate,
            "ReLU derivative at x={$x}: analytical={$analytical}, numerical={$numerical}, error={$error}",
            "ReLU derivative FAILED at x={$x}: error={$error} > {$tolerance}"
        );
    }
    
    return $result;
}

/**
 * Test activation function basic properties
 */
function test_activation_function_properties(): bool {
    $result = true;
    
    // Test that all activation functions implement the interface correctly
    $functions = [
        new Sigmoid(),
        new Tanh(),
        new ReLU(),
        new Softmax()
    ];
    
    foreach ($functions as $func) {
        // Test that getName() returns a non-empty string
        $name = $func->getName();
        $result = $result && Assert::True(
            !empty($name) && is_string($name),
            "Activation function {$name} has valid name",
            "Activation function name is invalid"
        );
        
        // Test that getRange() returns array with 2 elements
        $range = $func->getRange();
        $result = $result && Assert::True(
            is_array($range) && count($range) == 2,
            "Activation function {$name} has valid range format",
            "Activation function {$name} range format is invalid"
        );
        
        // Test that range values are numeric
        $result = $result && Assert::True(
            is_numeric($range[0]) && is_numeric($range[1]),
            "Activation function {$name} has numeric range values",
            "Activation function {$name} range contains non-numeric values"
        );
    }
    
    return $result;
}

/**
 * Test derivative function behavior at edge cases
 */
function test_derivative_edge_cases(): bool {
    $result = true;
    
    // Test Sigmoid at extreme values
    $sigmoid = new Sigmoid();
    $extremeValues = [-100, -10, 10, 100];
    
    foreach ($extremeValues as $x) {
        $derivative = $sigmoid->derivative($x);
        $result = $result && Assert::True(
            is_finite($derivative) && $derivative >= 0,
            "Sigmoid derivative at extreme value {$x} is finite and non-negative: {$derivative}",
            "Sigmoid derivative at extreme value {$x} is invalid: {$derivative}"
        );
    }
    
    // Test ReLU derivative discontinuity
    $relu = new ReLU();
    $result = $result && Assert::Equal(0.0, $relu->derivative(-0.1));
    $result = $result && Assert::Equal(0.0, $relu->derivative(0.0));
    $result = $result && Assert::Equal(1.0, $relu->derivative(0.1));
    
    return $result;
}

/**
 * Test mathematical relationship: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 */
function test_sigmoid_derivative_formula(): bool {
    $result = true;
    $sigmoid = new Sigmoid();
    
    $testValues = [-1.0, 0.0, 1.0];
    
    foreach ($testValues as $x) {
        $activation = $sigmoid->activate($x);
        $derivative = $sigmoid->derivative($x);
        $expected = $activation * (1.0 - $activation);
        
        $error = abs($derivative - $expected);
        $result = $result && Assert::True(
            $error < 1e-10,
            "Sigmoid derivative formula verified at x={$x}",
            "Sigmoid derivative formula FAILED at x={$x}: got {$derivative}, expected {$expected}"
        );
    }
    
    return $result;
}

/**
 * Test mathematical relationship: tanh'(x) = 1 - tanh²(x)
 */
function test_tanh_derivative_formula(): bool {
    $result = true;
    $tanh = new Tanh();
    
    $testValues = [-1.0, 0.0, 1.0];
    
    foreach ($testValues as $x) {
        $activation = $tanh->activate($x);
        $derivative = $tanh->derivative($x);
        $expected = 1.0 - ($activation * $activation);
        
        $error = abs($derivative - $expected);
        $result = $result && Assert::True(
            $error < 1e-10,
            "Tanh derivative formula verified at x={$x}",
            "Tanh derivative formula FAILED at x={$x}: got {$derivative}, expected {$expected}"
        );
    }
    
    return $result;
}

// Main test execution
echo "=== Testing Activation Function Derivatives for Backpropagation ===\n\n";

$framework = new Framework();
$framework->Reset();

// Add tests to framework
$framework->AddTest("test_sigmoid_derivative");
$framework->AddTest("test_tanh_derivative"); 
$framework->AddTest("test_relu_derivative");
$framework->AddTest("test_activation_function_properties");
$framework->AddTest("test_derivative_edge_cases");
$framework->AddTest("test_sigmoid_derivative_formula");
$framework->AddTest("test_tanh_derivative_formula");

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
    echo "\n🎉 All tests passed! Activation function derivatives are mathematically correct.\n";
} else {
    echo "\n❌ Some tests failed. Please check the implementation.\n";
}