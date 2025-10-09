<?php
/**
 * NiNa Neural Network Library - Quick Start Example
 * 
 * This example demonstrates the basic usage of the NiNa library
 * to create, train, and use a neural network.
 */

// Include all necessary files
include_once __DIR__ . '/../src/mc/lib.php';
include_once __DIR__ . '/../src/mc/nina/exception/NinaException.php';
include_once __DIR__ . '/../src/mc/nina/exception/ValidationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/ComputationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/SerializationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/LayerException.php';
include_once __DIR__ . '/../src/mc/nina/exception/NetworkException.php';
include_once __DIR__ . '/../src/mc/nina/exception/TrainingException.php';
include_once __DIR__ . '/../src/mc/nina/activation/ActivationFunction.php';
include_once __DIR__ . '/../src/mc/nina/activation/ReLU.php';
include_once __DIR__ . '/../src/mc/nina/activation/Sigmoid.php';
include_once __DIR__ . '/../src/mc/nina/activation/Tanh.php';
include_once __DIR__ . '/../src/mc/nina/activation/Softmax.php';
include_once __DIR__ . '/../src/mc/nina/cell.php';
include_once __DIR__ . '/../src/mc/nina/layer.php';
include_once __DIR__ . '/../src/mc/nina/network.php';
include_once __DIR__ . '/../src/mc/nina/trainer.php';
include_once __DIR__ . '/../src/mc/nina/predictor.php';

use mc\nina\Network;
use mc\nina\Layer;
use mc\nina\Cell;
use mc\nina\Trainer;
use mc\nina\Predictor;
use mc\nina\activation\Sigmoid;

echo "🚀 NiNa Quick Start Example\n";
echo "==========================\n\n";

try {
    // Step 1: Create neural network structure
    echo "Step 1: Creating neural network...\n";
    
    // Create a single cell for simplicity - 1 input, 1 output
    $cells = [
        new Cell("cell1", [0.5], new Sigmoid())   // 1 weight for 1 character input
    ];
    
    // Create layer with this cell
    $layer = new Layer("layer1", $cells);
    
    // Create the network
    $network = new Network("demo_network", [$layer]);
    echo "✓ Network created with 1 layer containing 1 cell (1 character input -> 1 character output)\n\n";
    
    // Step 2: Create trainer
    echo "Step 2: Setting up genetic algorithm trainer...\n";
    $trainer = new Trainer($network, 10); // Population size of 10
    echo "✓ Trainer configured with population size: 10\n\n";
    
    // Step 3: Train the network using string-based training
    echo "Step 3: Training network with sample data...\n";
    $trainer->Train("A", "X"); // Train to map single character "A" to "X"
    echo "✓ Training completed!\n\n";
    
    // Step 4: Get the best trained network
    echo "Step 4: Getting best network...\n";
    $bestNetwork = $trainer->getBestNetwork();
    echo "✓ Best network obtained\n\n";
    
    // Step 5: Test the network with calculations
    echo "Step 5: Testing the trained network...\n";
    $testInput = [0.255]; // Approximate value for character 'A'
    $result = $bestNetwork->Calculate($testInput);
    echo "  Input: [" . implode(", ", $testInput) . "] (representing 'A')\n";
    echo "  Output: [" . implode(", ", array_map(fn($x) => number_format($x, 3), $result)) . "]\n\n";
    
    // Step 6: Use Predictor for string-based predictions
    echo "Step 6: Using Predictor for easy predictions...\n";
    $predictor = new Predictor($bestNetwork);
    
    $prediction = $predictor->Predict('A');
    echo "  Predictor test: 'A' -> '" . $prediction . "'\n";
    $prediction2 = $predictor->Predict('B');
    echo "  Predictor test: 'B' -> '" . $prediction2 . "'\n\n";
    
    // Step 7: Save the trained model
    echo "Step 7: Saving the trained model...\n";
    $modelJson = $bestNetwork->Json();
    file_put_contents(__DIR__ . '/demo_model.json', $modelJson);
    echo "✓ Model saved to 'demo_model.json'\n\n";
    
    // Step 8: Show network information
    echo "Step 8: Network information...\n";
    echo "  Network name: " . $bestNetwork->GetName() . "\n";
    echo "  Number of layers: " . $bestNetwork->GetNrLayers() . "\n";
    echo "  Layer size: " . $bestNetwork->GetLayerSize() . "\n\n";
    
    echo "🎉 Quick start example completed successfully!\n";
    echo "You can now use the saved model or create new networks using NiNa.\n";
    
} catch (Exception $e) {
    echo "❌ Error: " . $e->getMessage() . "\n";
    echo "Stack trace:\n" . $e->getTraceAsString() . "\n";
}