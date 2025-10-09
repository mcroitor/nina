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
include_once __DIR__ . "/../src/mc/nina/exception/ComputationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/SerializationException.php";
include_once __DIR__ . "/../src/mc/nina/activation/ActivationFunction.php";
include_once __DIR__ . "/../src/mc/nina/activation/ReLU.php";
include_once __DIR__ . "/../src/mc/nina/activation/Sigmoid.php";
include_once __DIR__ . "/../src/mc/nina/activation/Tanh.php";
include_once __DIR__ . "/../src/mc/nina/cell.php";

use \mc\nina\Cell;
use \mc\nina\exception\ValidationException;
use \mc\nina\exception\ComputationException;
use \mc\nina\activation\ReLU;
use \mc\nina\activation\Sigmoid;
use \mc\nina\activation\Tanh;

function test_cell_1(): bool {
    $result = true;

    $cell_name = "sample cell";
    $dna_size = 100;
    $dna = [];
    $dna = array_pad($dna, $dna_size, 0.);
    $cell = new Cell($cell_name, $dna);

    $result = $result && Assert::Equal($cell_name, $cell->GetName());
    $result = $result && Assert::Equal($dna_size, $cell->GetSize());
    $result = $result && Assert::Equal($dna, $cell->GetDna());

    return $result;
}

function test_cell_2(): bool {
    $result = true;

    $cell_name = "sample cell";
    $dna_size = 100;

    $cell = Cell::Create($cell_name, $dna_size);

    $result = $result && Assert::Equal($cell_name, $cell->GetName());
    $result = $result && Assert::Equal($dna_size, $cell->GetSize());
    $result = $result && Assert::Equal(0., $cell->GetValue(0));
    $result = $result && Assert::Equal(0., $cell->GetValue($dna_size - 1));
    return $result;
}

function test_cell_3(): bool {
    $result = true;

    $cell_name = "sample cell";
    $dna_size = 100;
    $dna = \mc\array_create($dna_size, 0.5);
    $input = array_fill(0, $dna_size, 0.5);
    $expected_raw = $dna_size * 0.5 * 0.5; // Raw weighted sum = 25.0
    // With ReLU activation: max(0, 25.0) = 25.0

    $cell = new Cell($cell_name, $dna);
    $calculated = $cell->Calculate($input);
    $raw_calculated = $cell->CalculateRaw($input);

    $result = $result && Assert::Equal($expected_raw, $raw_calculated);
    $result = $result && Assert::Equal($expected_raw, $calculated); // ReLU doesn't change positive values

    return $result;
}

function test_cell_4(): bool {
    $result = true;
    $cell_name = "sample cell";
    $dna_size = 100;
    $dna = \mc\array_create($dna_size, 0.);
    $mutationPerCent = 10;

    $cell = new Cell($cell_name, $dna);
    $child = $cell->Born($mutationPerCent);

    $sum = array_sum($child->GetDna());
    $result = $result && Assert::NotEqual($sum, 0.);

    return $result;
}

function test_cell_5(): bool {
    $result = true;

    $cell_name = "sample cell";
    $dna_size = 100;
    $dna = \mc\array_create($dna_size, 0.5);
    
    $cell = new Cell($cell_name, $dna);

    $json = $cell->Json();

    $cell2 = Cell::Load($json);

    $result = $result && Assert::Equal($cell2->GetName(), $cell->GetName());
    $result = $result && Assert::Equal($cell2->GetDna(), $cell->GetDna());
    $result = $result && Assert::Equal($cell2->GetActivationFunction()->getName(), $cell->GetActivationFunction()->getName());
    return $result;
}

// NEW TESTS FOR ACTIVATION FUNCTIONS

function test_cell_activation_relu(): bool {
    $result = true;

    $cell_name = "relu_cell";
    $dna = [1.0, -1.0, 0.5, -0.5];
    $relu_activation = new ReLU();
    
    $cell = new Cell($cell_name, $dna, $relu_activation);
    
    // Test positive input (should remain unchanged)
    $positive_input = [1.0, 1.0, 1.0, 1.0];
    $positive_result = $cell->Calculate($positive_input); // 1*1 + (-1)*1 + 0.5*1 + (-0.5)*1 = 0
    $result = $result && Assert::Equal(0.0, $positive_result); // ReLU(0) = 0
    
    // Test input that produces positive weighted sum
    $input2 = [2.0, 0.0, 2.0, 0.0]; 
    $result2 = $cell->Calculate($input2); // 1*2 + (-1)*0 + 0.5*2 + (-0.5)*0 = 3.0
    $result = $result && Assert::Equal(3.0, $result2); // ReLU(3.0) = 3.0
    
    return $result;
}

function test_cell_activation_sigmoid(): bool {
    $result = true;

    $cell_name = "sigmoid_cell";
    $dna = [1.0];
    $sigmoid_activation = new Sigmoid();
    
    $cell = new Cell($cell_name, $dna, $sigmoid_activation);
    
    // Test zero input
    $zero_result = $cell->Calculate([0.0]);
    $result = $result && Assert::Equal(0.5, $zero_result); // sigmoid(0) = 0.5
    
    // Test that sigmoid is between 0 and 1
    $pos_result = $cell->Calculate([5.0]); // sigmoid(5) should be close to 1
    $neg_result = $cell->Calculate([-5.0]); // sigmoid(-5) should be close to 0
    
    $result = $result && ($pos_result > 0.9 && $pos_result < 1.0);
    $result = $result && ($neg_result > 0.0 && $neg_result < 0.1);
    
    return $result;
}

function test_cell_activation_tanh(): bool {
    $result = true;

    $cell_name = "tanh_cell";
    $dna = [1.0];
    $tanh_activation = new Tanh();
    
    $cell = new Cell($cell_name, $dna, $tanh_activation);
    
    // Test zero input
    $zero_result = $cell->Calculate([0.0]);
    $result = $result && Assert::Equal(0.0, $zero_result); // tanh(0) = 0
    
    // Test that tanh is between -1 and 1
    $pos_result = $cell->Calculate([5.0]); // tanh(5) should be close to 1
    $neg_result = $cell->Calculate([-5.0]); // tanh(-5) should be close to -1
    
    $result = $result && ($pos_result > 0.9 && $pos_result < 1.0);
    $result = $result && ($neg_result > -1.0 && $neg_result < -0.9);
    
    return $result;
}

function test_cell_change_activation(): bool {
    $result = true;

    $cell_name = "changeable_cell";
    $dna = [1.0];
    
    $cell = new Cell($cell_name, $dna); // Default ReLU
    
    // Test default activation
    $result = $result && Assert::Equal("relu", $cell->GetActivationFunction()->getName());
    
    // Change to Sigmoid
    $cell->SetActivationFunction(new Sigmoid());
    $result = $result && Assert::Equal("sigmoid", $cell->GetActivationFunction()->getName());
    
    // Test that activation actually changes the output
    $input = [-1.0]; // Negative weighted sum
    $sigmoid_result = $cell->Calculate($input);
    
    // Change back to ReLU
    $cell->SetActivationFunction(new ReLU());
    $relu_result = $cell->Calculate($input);
    
    // ReLU(-1) = 0, Sigmoid(-1) ≈ 0.27, so they should be different
    $result = $result && Assert::NotEqual($sigmoid_result, $relu_result);
    
    return $result;
}

// VALIDATION TESTS

function test_cell_validation_empty_name(): bool {
    $result = true;
    
    try {
        $cell = new Cell("", [1.0, 2.0]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::Equal("Invalid value for parameter 'name':  (expected non-empty string)", $e->getMessage());
    }
    
    return $result;
}

function test_cell_validation_empty_dna(): bool {
    $result = true;
    
    try {
        $cell = new Cell("test", []);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::Equal("Invalid value for parameter 'dna': empty array (expected non-empty array)", $e->getMessage());
    }
    
    return $result;
}

function test_cell_validation_non_numeric_dna(): bool {
    $result = true;
    
    try {
        $cell = new Cell("test", [1.0, "invalid", 3.0]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $expected_message = "Invalid value for parameter 'dna[1]': invalid (expected numeric) - got string";
        $result = $result && Assert::Equal($expected_message, $e->getMessage());
    }
    
    return $result;
}

function test_cell_validation_calculate_empty_input(): bool {
    $result = true;
    
    try {
        $cell = Cell::Create("test", 3);
        $cell->Calculate([]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::Equal("Invalid value for parameter 'input_values': empty array (expected non-empty array)", $e->getMessage());
    }
    
    return $result;
}

function test_cell_validation_calculate_size_mismatch(): bool {
    $result = true;
    
    try {
        $cell = Cell::Create("test", 3);
        $cell->Calculate([1.0, 2.0]); // Only 2 values for size 3
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $expected_message = "Invalid array size for 'input_values': expected 3, got 2";
        $result = $result && Assert::Equal($expected_message, $e->getMessage());
    }
    
    return $result;
}

function test_cell_validation_calculate_non_numeric_input(): bool {
    $result = true;
    
    try {
        $cell = Cell::Create("test", 3);
        $cell->Calculate([1.0, "invalid", 3.0]);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $expected_message = "Invalid value for parameter 'input_values[1]': invalid (expected numeric) - got string";
        $result = $result && Assert::Equal($expected_message, $e->getMessage());
    }
    
    return $result;
}

function test_cell_create_validation(): bool {
    $result = true;
    
    // Test invalid size
    try {
        $cell = Cell::Create("test", 0);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::Equal("Parameter 'size' value 0 is out of range (expected >= 1)", $e->getMessage());
    }
    
    // Test invalid default value
    try {
        $cell = Cell::Create("test", 5, INF);
        $result = false; // Should not reach here
    } catch (ComputationException $e) {
        $result = $result && Assert::Equal("Invalid numerical value (infinity) in default value", $e->getMessage());
    }
    
    return $result;
}

function test_cell_json_with_activation(): bool {
    $result = true;

    $cell_name = "json_test_cell";
    $dna = [0.1, 0.2, 0.3];
    $sigmoid = new Sigmoid();
    
    $cell = new Cell($cell_name, $dna, $sigmoid);
    
    $json = $cell->Json();
    $loaded_cell = Cell::Load($json);
    
    // Test that activation function is preserved
    $result = $result && Assert::Equal("sigmoid", $loaded_cell->GetActivationFunction()->getName());
    $result = $result && Assert::Equal($cell_name, $loaded_cell->GetName());
    $result = $result && Assert::Equal($dna, $loaded_cell->GetDna());
    
    // Test that calculation results are the same
    $input = [1.0, 1.0, 1.0];
    $original_result = $cell->Calculate($input);
    $loaded_result = $loaded_cell->Calculate($input);
    
    $result = $result && Assert::Equal($original_result, $loaded_result);
    
    return $result;
}

$framework = new Framework();

// Original tests
$framework->AddTest("test_cell_1");
$framework->AddTest("test_cell_2");
$framework->AddTest("test_cell_3");
$framework->AddTest("test_cell_4");
$framework->AddTest("test_cell_5");

// Activation function tests
$framework->AddTest("test_cell_activation_relu");
$framework->AddTest("test_cell_activation_sigmoid");
$framework->AddTest("test_cell_activation_tanh");
$framework->AddTest("test_cell_change_activation");

// Validation tests
$framework->AddTest("test_cell_validation_empty_name");
$framework->AddTest("test_cell_validation_empty_dna");
$framework->AddTest("test_cell_validation_non_numeric_dna");
$framework->AddTest("test_cell_validation_calculate_empty_input");
$framework->AddTest("test_cell_validation_calculate_size_mismatch");
$framework->AddTest("test_cell_validation_calculate_non_numeric_input");
$framework->AddTest("test_cell_create_validation");

// JSON serialization with activation test
$framework->AddTest("test_cell_json_with_activation");

$framework->Run();
$framework->PrintInfo();
