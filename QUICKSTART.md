# 🚀 Quick Start Guide - NiNa Neural Network Library

Welcome to NiNa! This guide will help you get started with the library in just a few minutes.

## 📋 Prerequisites

- PHP 8.0 or higher
- Basic understanding of neural networks
- Text editor or IDE

## 🔧 Installation

1. **Clone or download the library:**

   ```bash
   git clone https://github.com/yourusername/nina.git
   cd nina
   ```

2. **Include required files in your project:**

   ```php
   <?php
   // Core library
   include_once 'src/mc/lib.php';
   
   // Exception system
   include_once 'src/mc/nina/exception/NinaException.php';
   include_once 'src/mc/nina/exception/ValidationException.php';
   include_once 'src/mc/nina/exception/ComputationException.php';
   include_once 'src/mc/nina/exception/SerializationException.php';
   include_once 'src/mc/nina/exception/LayerException.php';
   include_once 'src/mc/nina/exception/NetworkException.php';
   include_once 'src/mc/nina/exception/TrainingException.php';
   
   // Activation functions
   include_once 'src/mc/nina/activation/ActivationFunction.php';
   include_once 'src/mc/nina/activation/ReLU.php';
   include_once 'src/mc/nina/activation/Sigmoid.php';
   include_once 'src/mc/nina/activation/Tanh.php';
   include_once 'src/mc/nina/activation/Softmax.php';
   
   // Neural network components
   include_once 'src/mc/nina/cell.php';
   include_once 'src/mc/nina/layer.php';
   include_once 'src/mc/nina/network.php';
   include_once 'src/mc/nina/trainer.php';
   include_once 'src/mc/nina/predictor.php';
   
   // Use the library
   use mc\nina\Cell;
   use mc\nina\Layer;
   use mc\nina\Network;
   use mc\nina\Trainer;
   use mc\nina\Predictor;
   use mc\nina\activation\Sigmoid;
   use mc\nina\activation\ReLU;
   ```

## 🎯 Your First Neural Network

### Step 1: Create Individual Cells (Neurons)

```php
// Create cells with random weights
$cell1 = Cell::Create("input1", 2, 0.5);  // 2 weights, initialized to 0.5
$cell2 = Cell::Create("input2", 2, 0.3);  // 2 weights, initialized to 0.3
$output_cell = Cell::Create("output", 2, 0.8);  // Output neuron

// Or create with specific weights
$cell_with_weights = new Cell("custom", [0.2, 0.7, 0.4], new Sigmoid());
```

### Step 2: Create Layers

```php
// Create input layer with 2 neurons
$input_layer = new Layer("input_layer", [$cell1, $cell2]);

// Create output layer with 1 neuron
$output_layer = new Layer("output_layer", [$output_cell]);

// Or use the quick create method
$hidden_layer = Layer::Create(3);  // Creates layer with 3 random neurons
```

### Step 3: Build Your Network

```php
// Combine layers into a network
$network = new Network([$input_layer, $output_layer]);

// Test the network
$input_data = [1.0, 0.5];  // Input values
$result = $network->Calculate($input_data);
echo "Network output: " . $result . "\n";
```

## 🎓 Train Your Network

### Using Genetic Algorithm Training

```php
// Create a trainer with population of 20 networks
$trainer = new Trainer($network, 20);

// Training data: input -> expected output
$training_data = [
    ["input" => [1.0, 0.0], "expected" => "positive"],
    ["input" => [0.0, 1.0], "expected" => "negative"],
    ["input" => [1.0, 1.0], "expected" => "positive"],
    ["input" => [0.0, 0.0], "expected" => "negative"]
];

// Train the network
echo "Training network...\n";
foreach ($training_data as $data) {
    $input_string = implode('', array_map('strval', $data["input"]));
    $trainer->Train($input_string, $data["expected"]);
}

// Get the best trained network
$best_network = $trainer->getBestNetwork();
echo "Training completed!\n";
```

## 🔮 Make Predictions

```php
// Create a predictor with the trained network
$predictor = new Predictor($best_network);

// Make predictions
$test_inputs = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
];

foreach ($test_inputs as $input) {
    $input_string = implode('', array_map('strval', $input));
    $prediction = $predictor->Predict($input_string);
    echo "Input: [" . implode(', ', $input) . "] -> Prediction: " . $prediction . "\n";
}
```

## 🧠 Complete Example: XOR Problem

Here's a complete example solving the classic XOR problem:

```php
<?php
// Include all necessary files (as shown above)

try {
    // Create XOR network: 2 inputs -> 4 hidden -> 1 output
    $input_layer = Layer::Create(2);     // 2 input neurons
    $hidden_layer = Layer::Create(4);    // 4 hidden neurons  
    $output_layer = Layer::Create(1);    // 1 output neuron
    
    $network = new Network([$input_layer, $hidden_layer, $output_layer]);
    
    // Create trainer
    $trainer = new Trainer($network, 50);  // Population of 50
    
    // XOR training data
    $xor_data = [
        ["00", "0"],  // 0 XOR 0 = 0
        ["01", "1"],  // 0 XOR 1 = 1
        ["10", "1"],  // 1 XOR 0 = 1
        ["11", "0"]   // 1 XOR 1 = 0
    ];
    
    // Train for multiple epochs
    echo "Training XOR network...\n";
    for ($epoch = 0; $epoch < 100; $epoch++) {
        foreach ($xor_data as $data) {
            $trainer->Train($data[0], $data[1]);
        }
        if ($epoch % 20 == 0) {
            echo "Epoch $epoch completed\n";
        }
    }
    
    // Test the trained network
    $predictor = new Predictor($trainer->getBestNetwork());
    
    echo "\nXOR Results:\n";
    foreach ($xor_data as $data) {
        $prediction = $predictor->Predict($data[0]);
        echo "Input: {$data[0]} -> Expected: {$data[1]} -> Got: $prediction\n";
    }
    
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
    echo "Code: " . $e->getCode() . "\n";
}
```

## 🎛️ Activation Functions

NiNa supports multiple activation functions:

```php
use mc\nina\activation\Sigmoid;
use mc\nina\activation\ReLU;
use mc\nina\activation\Tanh;
use mc\nina\activation\Softmax;

// Create cells with different activation functions
$sigmoid_cell = new Cell("sigmoid", [0.5, 0.3], new Sigmoid());
$relu_cell = new Cell("relu", [0.4, 0.6], new ReLU());
$tanh_cell = new Cell("tanh", [0.2, 0.8], new Tanh());

// Change activation function on existing cell
$cell = Cell::Create("test", 3);
$cell->SetActivationFunction(new Sigmoid());
```

## 💾 Save and Load Models

```php
// Save your trained network
$network_json = $best_network->Json();
file_put_contents("my_model.json", $network_json);

// Load the network later
$loaded_json = file_get_contents("my_model.json");
$loaded_network = Network::Load($loaded_json);

// Use the loaded network
$predictor = new Predictor($loaded_network);
$result = $predictor->Predict("test_input");
```

## 🛡️ Error Handling

NiNa has a robust exception system:

```php
use mc\nina\exception\ValidationException;
use mc\nina\exception\TrainingException;
use mc\nina\exception\ComputationException;

try {
    // Your neural network code
    $network = new Network($layers);
    $result = $network->Calculate($input);
    
} catch (ValidationException $e) {
    echo "Validation Error: " . $e->getMessage() . "\n";
    echo "Context: " . json_encode($e->getContext()) . "\n";
    
} catch (TrainingException $e) {
    echo "Training Error: " . $e->getMessage() . "\n";
    
} catch (ComputationException $e) {
    echo "Computation Error: " . $e->getMessage() . "\n";
    
} catch (Exception $e) {
    echo "General Error: " . $e->getMessage() . "\n";
}
```

## 📁 Project Structure

```
your_project/
├── nina/                    # NiNa library
│   ├── src/
│   │   └── mc/
│   │       ├── lib.php
│   │       └── nina/
│   │           ├── activation/
│   │           ├── exception/
│   │           ├── cell.php
│   │           ├── layer.php
│   │           ├── network.php
│   │           ├── trainer.php
│   │           └── predictor.php
│   └── examples/
├── my_neural_network.php    # Your main script
└── models/                  # Saved models
    └── my_model.json
```

## 🎯 Next Steps

1. **Explore Examples**: Check the `examples/` directory for more complex scenarios
2. **Read the Documentation**: See `README.md` and `API.md` for detailed information  
3. **Experiment**: Try different activation functions and network architectures
4. **Join the Community**: Contribute to the project or report issues

## 🔗 Useful Links

- [Full API Documentation](API.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Exception Handling Guide](EXCEPTIONS.md)
- [Examples Directory](examples/)
- [Russian Documentation](README_RU.md)

## 🆘 Common Issues

**Problem**: "Class not found" errors  
**Solution**: Make sure all required files are included in the correct order

**Problem**: Training doesn't improve  
**Solution**: Try larger population size, more training epochs, or different network architecture

**Problem**: Memory issues with large networks  
**Solution**: Reduce population size or network complexity

**Problem**: Validation errors  
**Solution**: Check input data types and ranges - all inputs should be numeric

---

**Happy Neural Networking! 🧠✨**

For more help, check the full documentation or create an issue on GitHub.