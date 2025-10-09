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
include_once __DIR__ . "/../src/mc/nina/exception/LayerException.php";
include_once __DIR__ . "/../src/mc/nina/activation/ActivationFunction.php";
include_once __DIR__ . "/../src/mc/nina/activation/ReLU.php";
include_once __DIR__ . "/../src/mc/nina/activation/Sigmoid.php";
include_once __DIR__ . "/../src/mc/nina/activation/Tanh.php";
include_once __DIR__ . "/../src/mc/nina/cell.php";
include_once __DIR__ . "/../src/mc/nina/layer.php";

use \mc\nina\Cell;
use \mc\nina\Layer;
use \mc\nina\exception\ValidationException;
use \mc\nina\exception\LayerException;
use \mc\nina\activation\ReLU;
use \mc\nina\activation\Sigmoid;
use \mc\nina\activation\Tanh;

// BASIC LAYER TESTS

function test_layer_constructor(): bool {
    $result = true;
    
    $layer_name = "test_layer";
    $cells = [
        new Cell("cell1", [0.1, 0.2, 0.3]),
        new Cell("cell2", [0.4, 0.5, 0.6]),
        new Cell("cell3", [0.7, 0.8, 0.9])
    ];
    
    $layer = new Layer($layer_name, $cells);
    
    $result = $result && Assert::Equal($layer_name, $layer->GetName());
    $result = $result && Assert::Equal(3, $layer->GetSize());
    $result = $result && Assert::Equal($cells[0], $layer->GetCell(0));
    $result = $result && Assert::Equal($cells[1], $layer->GetCell(1));
    $result = $result && Assert::Equal($cells[2], $layer->GetCell(2));
    
    return $result;
}

function test_layer_create(): bool {
    $result = true;
    
    $size = 5;
    $layer = Layer::Create($size);
    
    $result = $result && Assert::Equal($size, $layer->GetSize());
    
    // Check that all cells have the same size
    for ($i = 0; $i < $size; $i++) {
        $cell = $layer->GetCell($i);
        $result = $result && Assert::Equal($size, $cell->GetSize());
        // Check that cell name is not empty
        $result = $result && (!empty($cell->GetName()));
    }
    
    return $result;
}

function test_layer_calculate(): bool {
    $result = true;
    
    // Create layer with known values
    $cells = [
        new Cell("cell1", [1.0, 0.0, 0.0]), // Will output input[0]
        new Cell("cell2", [0.0, 1.0, 0.0]), // Will output input[1] 
        new Cell("cell3", [0.0, 0.0, 1.0])  // Will output input[2]
    ];
    
    $layer = new Layer("identity_layer", $cells);
    
    $input = [2.0, 3.0, 4.0];
    $output = $layer->Calculate($input);
    
    // With ReLU activation (default), positive values pass through
    $result = $result && Assert::Equal(3, count($output));
    $result = $result && Assert::Equal(2.0, $output[0]);
    $result = $result && Assert::Equal(3.0, $output[1]);
    $result = $result && Assert::Equal(4.0, $output[2]);
    
    return $result;
}

function test_layer_born(): bool {
    $result = true;
    
    $original_layer = Layer::Create(3);
    $child_layer = $original_layer->Born(10);
    
    // Child should have same structure
    $result = $result && Assert::Equal($original_layer->GetSize(), $child_layer->GetSize());
    
    // But different DNA values (due to mutation)
    $different_found = false;
    for ($i = 0; $i < $original_layer->GetSize(); $i++) {
        $original_dna = $original_layer->GetCell($i)->GetDna();
        $child_dna = $child_layer->GetCell($i)->GetDna();
        
        for ($j = 0; $j < count($original_dna); $j++) {
            if ($original_dna[$j] !== $child_dna[$j]) {
                $different_found = true;
                break 2;
            }
        }
    }
    
    $result = $result && $different_found;
    
    return $result;
}

function test_layer_json_serialization(): bool {
    $result = true;
    
    $layer_name = "json_test_layer";
    $cells = [
        new Cell("cell1", [0.1, 0.2], new Sigmoid()),
        new Cell("cell2", [0.3, 0.4], new Tanh())
    ];
    
    $layer = new Layer($layer_name, $cells);
    
    $json = $layer->Json();
    $loaded_layer = Layer::Load($json);
    
    $result = $result && Assert::Equal($layer_name, $loaded_layer->GetName());
    $result = $result && Assert::Equal($layer->GetSize(), $loaded_layer->GetSize());
    
    // Check that cells are properly restored
    for ($i = 0; $i < $layer->GetSize(); $i++) {
        $original_cell = $layer->GetCell($i);
        $loaded_cell = $loaded_layer->GetCell($i);
        
        $result = $result && Assert::Equal($original_cell->GetName(), $loaded_cell->GetName());
        $result = $result && Assert::Equal($original_cell->GetDna(), $loaded_cell->GetDna());
        // Note: Current implementation doesn't preserve activation functions in Layer::Load
        // This is a limitation that could be improved
    }
    
    return $result;
}

// VALIDATION TESTS

function test_layer_validation_empty_name(): bool {
    $result = true;
    
    try {
        $cells = [new Cell("cell1", [1.0, 2.0])];
        $layer = new Layer("", $cells);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $result = $result && Assert::Equal("Invalid value for parameter 'name':  (expected non-empty string)", $e->getMessage());
    }
    
    return $result;
}

function test_layer_validation_empty_cells(): bool {
    $result = true;
    
    try {
        $layer = new Layer("test_layer", []);
        $result = false; // Should not reach here
    } catch (LayerException $e) {
        $result = $result && Assert::Equal("Invalid layer size: 0 - layer must contain at least one cell", $e->getMessage());
    }
    
    return $result;
}

function test_layer_validation_invalid_cell_type(): bool {
    $result = true;
    
    try {
        $cells = [
            new Cell("cell1", [1.0, 2.0]),
            "not_a_cell", // Invalid type
            new Cell("cell3", [5.0, 6.0])
        ];
        $layer = new Layer("test_layer", $cells);
        $result = false; // Should not reach here
    } catch (ValidationException $e) {
        $expected_message = "Invalid value for parameter 'cells[1]': not_a_cell (expected Cell instance) - got string";
        $result = $result && Assert::Equal($expected_message, $e->getMessage());
    }
    
    return $result;
}

function test_layer_create_validation(): bool {
    $result = true;
    
    try {
        $layer = Layer::Create(0);
        $result = false; // Should not reach here
    } catch (LayerException $e) {
        $result = $result && Assert::Equal("Invalid layer size: 0 - layer size must be positive", $e->getMessage());
    }
    
    // Test negative size
    try {
        $layer = Layer::Create(-5);
        $result = false; // Should not reach here
    } catch (LayerException $e) {
        $result = $result && Assert::Equal("Invalid layer size: -5 - layer size must be positive", $e->getMessage());
    }
    
    return $result;
}

// ACTIVATION FUNCTION INTEGRATION TESTS

function test_layer_mixed_activations(): bool {
    $result = true;
    
    // Create layer with different activation functions
    $cells = [
        new Cell("relu_cell", [1.0, -1.0], new ReLU()),
        new Cell("sigmoid_cell", [0.0, 1.0], new Sigmoid()),
        new Cell("tanh_cell", [1.0, 0.0], new Tanh())
    ];
    
    $layer = new Layer("mixed_layer", $cells);
    
    $input = [-1.0, 1.0]; // Negative and positive values
    $output = $layer->Calculate($input);
    
    // ReLU cell: 1*(-1) + (-1)*1 = -2, ReLU(-2) = 0
    $result = $result && Assert::Equal(0.0, $output[0]);
    
    // Sigmoid cell: 0*(-1) + 1*1 = 1, Sigmoid(1) ≈ 0.73
    $result = $result && ($output[1] > 0.7 && $output[1] < 0.8);
    
    // Tanh cell: 1*(-1) + 0*1 = -1, Tanh(-1) ≈ -0.76
    $result = $result && ($output[2] > -0.8 && $output[2] < -0.7);
    
    return $result;
}

function test_layer_uniform_activation(): bool {
    $result = true;
    
    // Create layer where all cells use the same activation
    $sigmoid = new Sigmoid();
    $cells = [
        new Cell("cell1", [1.0, 0.0], $sigmoid),
        new Cell("cell2", [0.0, 1.0], $sigmoid)
    ];
    
    $layer = new Layer("sigmoid_layer", $cells);
    
    $input = [0.0, 0.0]; // Zero input should give sigmoid(0) = 0.5 for both
    $output = $layer->Calculate($input);
    
    $result = $result && Assert::Equal(2, count($output));
    $result = $result && Assert::Equal(0.5, $output[0]);
    $result = $result && Assert::Equal(0.5, $output[1]);
    
    return $result;
}

// EDGE CASE TESTS

function test_layer_single_cell(): bool {
    $result = true;
    
    $cell = new Cell("single_cell", [2.0, 3.0]);
    $layer = new Layer("single_layer", [$cell]);
    
    $result = $result && Assert::Equal(1, $layer->GetSize());
    $result = $result && Assert::Equal($cell, $layer->GetCell(0));
    
    $input = [1.0, 1.0];
    $output = $layer->Calculate($input);
    
    $result = $result && Assert::Equal(1, count($output));
    $result = $result && Assert::Equal(5.0, $output[0]); // 2*1 + 3*1 = 5, ReLU(5) = 5
    
    return $result;
}

function test_layer_large_size(): bool {
    $result = true;
    
    $size = 100;
    $layer = Layer::Create($size);
    
    $result = $result && Assert::Equal($size, $layer->GetSize());
    
    // Test that we can calculate with large layer
    $input = array_fill(0, $size, 0.1); // All 0.1
    $output = $layer->Calculate($input);
    
    $result = $result && Assert::Equal($size, count($output));
    
    // All outputs should be non-negative (ReLU)
    foreach ($output as $value) {
        $result = $result && ($value >= 0.0);
    }
    
    return $result;
}

function test_layer_calculate_with_different_cell_sizes(): bool {
    $result = true;
    
    // All cells in a layer should have the same DNA size for proper calculation
    $cells = [
        new Cell("cell1", [1.0, 2.0, 3.0]),
        new Cell("cell2", [4.0, 5.0, 6.0]),
        new Cell("cell3", [7.0, 8.0, 9.0])
    ];
    
    $layer = new Layer("consistent_layer", $cells);
    
    $input = [1.0, 1.0, 1.0];
    $output = $layer->Calculate($input);
    
    // Expected outputs: 
    // cell1: 1*1 + 2*1 + 3*1 = 6, ReLU(6) = 6
    // cell2: 4*1 + 5*1 + 6*1 = 15, ReLU(15) = 15
    // cell3: 7*1 + 8*1 + 9*1 = 24, ReLU(24) = 24
    
    $result = $result && Assert::Equal(3, count($output));
    $result = $result && Assert::Equal(6.0, $output[0]);
    $result = $result && Assert::Equal(15.0, $output[1]);
    $result = $result && Assert::Equal(24.0, $output[2]);
    
    return $result;
}

// PERFORMANCE AND STRESS TESTS

function test_layer_born_preserves_structure(): bool {
    $result = true;
    
    $original = Layer::Create(5);
    
    // Test multiple generations
    $current = $original;
    for ($i = 0; $i < 10; $i++) {
        $current = $current->Born(5);
        
        // Structure should be preserved
        $result = $result && Assert::Equal($original->GetSize(), $current->GetSize());
        
        // Each cell should have same size DNA
        for ($j = 0; $j < $current->GetSize(); $j++) {
            $result = $result && Assert::Equal(
                $original->GetCell($j)->GetSize(),
                $current->GetCell($j)->GetSize()
            );
        }
    }
    
    return $result;
}

$framework = new Framework();

// Basic functionality tests
$framework->AddTest("test_layer_constructor");
$framework->AddTest("test_layer_create");
$framework->AddTest("test_layer_calculate");
$framework->AddTest("test_layer_born");
$framework->AddTest("test_layer_json_serialization");

// Validation tests
$framework->AddTest("test_layer_validation_empty_name");
$framework->AddTest("test_layer_validation_empty_cells");
$framework->AddTest("test_layer_validation_invalid_cell_type");
$framework->AddTest("test_layer_create_validation");

// Activation function integration tests
$framework->AddTest("test_layer_mixed_activations");
$framework->AddTest("test_layer_uniform_activation");

// Edge case tests
$framework->AddTest("test_layer_single_cell");
$framework->AddTest("test_layer_large_size");
$framework->AddTest("test_layer_calculate_with_different_cell_sizes");

// Performance tests
$framework->AddTest("test_layer_born_preserves_structure");

$framework->Run();
$framework->PrintInfo();

?>