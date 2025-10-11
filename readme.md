# NiNa - Neural Network Library for PHP

![PHP Version](https://img.shields.io/badge/PHP-8.0%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

**NiNa** is a powerful yet simple PHP neural network library designed for both beginners and advanced users. It provides a clean, object-oriented API for creating, training, and using neural networks with support for multiple activation functions and comprehensive error handling.

## ✨ Features

- 🧠 **Multiple Activation Functions**: Sigmoid, Tanh, ReLU, Softmax
- 🛡️ **Robust Error Handling**: Comprehensive exception system with detailed error context
- 🧪 **Well Tested**: Extensive test suite with 100% pass rate
- 🔧 **Easy to Use**: Clean, intuitive API
- 📊 **Genetic Algorithm**: Built-in evolutionary training approach
- 💾 **Serialization**: Save and load trained models
- 🎯 **Flexible Architecture**: Customizable network structures

## 📦 Installation

### Manual Installation

Clone the repository and include the library:

```php
<?php
require_once 'src/mc/lib.php';
require_once 'src/mc/nina/cell.php';
require_once 'src/mc/nina/layer.php';
require_once 'src/mc/nina/network.php';
require_once 'src/mc/nina/trainer.php';
require_once 'src/mc/nina/predictor.php';

// Include activation functions
require_once 'src/mc/nina/activation/ActivationFunction.php';
require_once 'src/mc/nina/activation/ReLU.php';
require_once 'src/mc/nina/activation/Sigmoid.php';
// ... other activations
```

## 🚀 Quick Start

### Basic Neural Network Creation

```php
<?php
use mc\nina\Network;
use mc\nina\Trainer;
use mc\nina\Predictor;

// Create a simple 3-layer network with 5 neurons each
$network = Network::Create("MyNetwork", 3, 5);

echo "Created network: " . $network->GetName() . PHP_EOL;
echo "Layers: " . $network->GetNrLayers() . PHP_EOL;
echo "Layer size: " . $network->GetLayerSize() . PHP_EOL;
```

### Training with Genetic Algorithm

```php
<?php
// Create trainer with population of 50 networks
$trainer = new Trainer($network, 50);

// Training data
$input = "hello";
$expected = "world";

// Train the network
$trainer->Train($input, $expected);

// Get the best performing network
$bestNetwork = $trainer->getBestNetwork();
```

### Making Predictions

```php
<?php
// Create predictor with trained network
$predictor = new Predictor($bestNetwork);

// Make prediction
$input = "hello";
$prediction = $predictor->Predict($input);

echo "Input: {$input}" . PHP_EOL;
echo "Prediction: {$prediction}" . PHP_EOL;
```

## 🧠 Activation Functions

NiNa supports multiple activation functions that can be applied to individual cells:

### Available Activation Functions

```php
<?php
use mc\nina\activation\{ReLU, Sigmoid, Tanh, Softmax};
use mc\nina\Cell;

// Create cells with different activation functions
$reluCell = new Cell("relu_cell", [0.5, -0.2, 0.8], new ReLU());
$sigmoidCell = new Cell("sigmoid_cell", [0.3, 0.7, -0.1], new Sigmoid());
$tanhCell = new Cell("tanh_cell", [0.1, -0.4, 0.6], new Tanh());
$softmaxCell = new Cell("softmax_cell", [0.2, 0.5, 0.3], new Softmax());

// Calculate outputs
$input = [1.0, 0.5, -0.3];
echo "ReLU output: " . $reluCell->Calculate($input) . PHP_EOL;
echo "Sigmoid output: " . $sigmoidCell->Calculate($input) . PHP_EOL;
echo "Tanh output: " . $tanhCell->Calculate($input) . PHP_EOL;
echo "Softmax output: " . $softmaxCell->Calculate($input) . PHP_EOL;
```

### Activation Function Properties

| Function | Range | Derivative | Use Case |
|----------|-------|------------|----------|
| **ReLU** | [0, ∞) | 1 if x > 0, else 0 | Hidden layers, faster training |
| **Sigmoid** | (0, 1) | σ(x)(1-σ(x)) | Binary classification, output layers |
| **Tanh** | (-1, 1) | 1-tanh²(x) | Hidden layers, centered output |
| **Softmax** | (0, 1) | Complex | Multi-class classification |

## 🏗️ Advanced Usage

### Custom Network Architecture

```php
<?php
use mc\nina\{Cell, Layer, Network};
use mc\nina\activation\{ReLU, Sigmoid};

// Create custom cells with specific weights and activations
$inputCells = [
    new Cell("input_1", [0.5, 0.3, -0.2], new ReLU()),
    new Cell("input_2", [0.1, 0.8, 0.4], new ReLU()),
    new Cell("input_3", [-0.3, 0.6, 0.2], new ReLU())
];

$hiddenCells = [
    new Cell("hidden_1", [0.7, -0.1, 0.4], new Sigmoid()),
    new Cell("hidden_2", [0.2, 0.5, -0.6], new Sigmoid())
];

$outputCells = [
    new Cell("output_1", [0.3, -0.4], new Sigmoid())
];

// Create layers
$inputLayer = new Layer("InputLayer", $inputCells);
$hiddenLayer = new Layer("HiddenLayer", $hiddenCells);
$outputLayer = new Layer("OutputLayer", $outputCells);

// Create network from layers
$customNetwork = new Network("CustomNetwork", [$inputLayer, $hiddenLayer, $outputLayer]);
```

### Error Handling

NiNa provides comprehensive error handling with specific exception types:

```php
<?php
use mc\nina\exception\{
    ValidationException,
    NetworkException,
    LayerException,
    TrainingException,
    SerializationException,
    ComputationException
};

try {
    // Attempt to create network with invalid parameters
    $network = Network::Create("", 0, -5);
} catch (ValidationException $e) {
    echo "Validation error: " . $e->getMessage() . PHP_EOL;
    echo "Error code: " . $e->getCode() . PHP_EOL;
    echo "Context: " . json_encode($e->getContext()) . PHP_EOL;
} catch (NetworkException $e) {
    echo "Network architecture error: " . $e->getMessage() . PHP_EOL;
}

try {
    // Attempt computation with invalid data
    $cell = new Cell("test", [1.0, 2.0]);
    $result = $cell->Calculate([INF, NAN]); // Will throw ComputationException
} catch (ComputationException $e) {
    echo "Computation error: " . $e->getDetailedMessage() . PHP_EOL;
}
```

### Model Serialization

```php
<?php
// Save trained network
$networkJson = $bestNetwork->Json();
file_put_contents('trained_model.json', $networkJson);

// Load network from file
$loadedJson = file_get_contents('trained_model.json');
$loadedNetwork = Network::Load($loadedJson);

// Verify loaded network
echo "Original network layers: " . $bestNetwork->GetNrLayers() . PHP_EOL;
echo "Loaded network layers: " . $loadedNetwork->GetNrLayers() . PHP_EOL;
```

## 🧪 Testing and Validation

### Running Tests

```bash
# Run all tests
php tests/test_cell.php
php tests/test_layer.php
php tests/test_exceptions.php

# Example output:
# =============== TESTS ================
# total: 15, passed: 15, failed: 0
# ============== ASSERTS ===============
# total: 97, passed: 97, failed: 0
```

### Example Training Loop

```php
<?php
function trainXORNetwork() {
    $network = Network::Create("XOR_Network", 3, 4);
    $trainer = new Trainer($network, 100);
    
    $trainingData = [
        ['00', '0'],
        ['01', '1'],
        ['10', '1'],
        ['11', '0']
    ];
    
    echo "Training XOR network..." . PHP_EOL;
    
    for ($epoch = 0; $epoch < 1000; $epoch++) {
        foreach ($trainingData as $data) {
            $trainer->Train($data[0], $data[1]);
        }
        
        if ($epoch % 100 == 0) {
            echo "Epoch {$epoch} completed" . PHP_EOL;
        }
    }
    
    $bestNetwork = $trainer->getBestNetwork();
    $predictor = new Predictor($bestNetwork);
    
    echo "Testing trained network:" . PHP_EOL;
    foreach ($trainingData as $data) {
        $prediction = $predictor->Predict($data[0]);
        echo "Input: {$data[0]} -> Expected: {$data[1]}, Predicted: {$prediction}" . PHP_EOL;
    }
    
    return $bestNetwork;
}

// Run the training
$xorNetwork = trainXORNetwork();
```

## 🎯 Use Cases

### 1. Pattern Recognition

```php
<?php
// Train network to recognize patterns in strings
$patterns = [
    ['pattern_a', 'type_1'],
    ['pattern_b', 'type_2'],
    ['pattern_c', 'type_1']
];

$network = Network::Create("PatternRecognizer", 2, 10);
$trainer = new Trainer($network, 50);

foreach ($patterns as $pattern) {
    $trainer->Train($pattern[0], $pattern[1]);
}
```

### 2. Function Approximation

```php
<?php
// Approximate mathematical functions
function trainFunctionApproximator() {
    $network = Network::Create("FunctionApproximator", 3, 8);
    $trainer = new Trainer($network, 75);
    
    // Generate training data for f(x) = x^2
    for ($i = 0; $i < 100; $i++) {
        $x = mt_rand(-100, 100) / 10.0; // Random number -10 to 10
        $y = $x * $x; // f(x) = x^2
        
        $trainer->Train((string)$x, (string)$y);
    }
    
    return $trainer->getBestNetwork();
}
```

### 3. Data Classification

```php
<?php
// Simple binary classifier
function createBinaryClassifier() {
    $network = Network::Create("BinaryClassifier", 2, 6);
    $trainer = new Trainer($network, 40);
    
    // Training data: classify as positive (1) or negative (0)
    $data = [
        ['positive_example_1', '1'],
        ['positive_example_2', '1'],
        ['negative_example_1', '0'],
        ['negative_example_2', '0']
    ];
    
    foreach ($data as $sample) {
        $trainer->Train($sample[0], $sample[1]);
    }
    
    return $trainer->getBestNetwork();
}
```

## 🛠️ Development and Contributing

### Project Structure

```
nina/
├── src/mc/
│   ├── lib.php                    # Utility functions
│   └── nina/
│       ├── cell.php               # Neural cell implementation
│       ├── layer.php              # Layer implementation
│       ├── network.php            # Network implementation
│       ├── trainer.php            # Genetic training algorithm
│       ├── predictor.php          # Prediction interface
│       ├── activation/            # Activation functions
│       │   ├── ActivationFunction.php
│       │   ├── ReLU.php
│       │   ├── Sigmoid.php
│       │   ├── Tanh.php
│       │   └── Softmax.php
│       └── exception/             # Exception system
│           ├── NinaException.php
│           ├── ValidationException.php
│           ├── NetworkException.php
│           ├── LayerException.php
│           ├── TrainingException.php
│           ├── SerializationException.php
│           └── ComputationException.php
├── tests/                         # Test suites
├── docs/                          # Documentation
└── README.md
```

### Exception System

NiNa uses a comprehensive exception system for better error handling:

- **ValidationException** (6001-6099): Input validation errors
- **NetworkException** (1001-1099): Network architecture issues
- **LayerException** (2001-2099): Layer configuration problems
- **TrainingException** (3001-3099): Training process errors
- **SerializationException** (4001-4099): Save/load operations
- **ComputationException** (5001-5099): Mathematical computation errors

For detailed exception documentation, see [docs/exceptions.md](docs/exceptions.md).

## 🔗 API Reference

### Core Classes

- **`Network`**: Main neural network container
- **`Layer`**: Collection of neural cells
- **`Cell`**: Individual neuron with weights and activation
- **`Trainer`**: Genetic algorithm training implementation
- **`Predictor`**: Interface for making predictions

### Key Methods

```php
// Network creation and management
Network::Create(string $name, int $layers, int $layerSize): Network
$network->Calculate(array $input): array
$network->Json(): string
Network::Load(string $json): Network

// Training and evolution
new Trainer(Network $network, int $populationSize)
$trainer->Train(string $input, string $expected): void
$trainer->getBestNetwork(): Network

// Prediction
new Predictor(Network $network)
$predictor->Predict(string $input): string

// Cell operations
new Cell(string $name, array $weights, ActivationFunction $activation)
$cell->Calculate(array $input): float
$cell->Born(int $mutationPercent): Cell
```

## 📚 Additional Resources

- **[Complete Documentation](docs/)** - Full documentation index
- **[API Reference](docs/api.md)** - Detailed API documentation with examples
- **[Architecture Guide](docs/architecture.md)** - System design and patterns
- **[Exception System](docs/exceptions.md)** - Comprehensive guide to error handling
- **[Genetic Algorithm](docs/genetic_algorithm.md)** - Complete guide to evolutionary neural network training
- **[Russian Documentation](docs/ru/)** - Русскоязычная документация
- **[Test Suite](tests/)** - Examples and test cases
- **[Source Code](src/)** - Implementation details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Roadmap

- ✅ **v0.2.0**: Activation functions and exception system (COMPLETED)
- 🔄 **v0.3.0**: Backpropagation and loss functions (IN PROGRESS)
- 📋 **v0.4.0**: Advanced architectures and metrics
- 🎯 **v1.0.0**: Production-ready release

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for simple, educational neural networks in PHP
- Built with focus on clarity and ease of understanding
- Designed for both learning and practical applications

---

**Happy Neural Networking! 🧠✨**
