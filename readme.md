# NiNa - Neural Network Library for PHP

![PHP Version](https://img.shields.io/badge/PHP-8.0%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

**NiNa** is a lightweight, object-oriented neural network library written in pure PHP. It provides full support for both gradient-based **Backpropagation** (with SGD, Adam, and RMSprop optimizers) and evolutionary **Genetic Algorithm** training.

---

## Features

- **Dual Training Paradigms**: Backpropagation (gradient descent) and Genetic Algorithm (evolutionary optimization).
- **Modern Optimizers**: SGD (with momentum and Nesterov), Adam, and RMSprop.
- **Multiple Activation Functions**: Sigmoid, Tanh, ReLU, and Softmax with verified analytical derivatives.
- **Loss Functions**: Mean Squared Error (MSE) and Cross-Entropy.
- **Learning Rate Scheduling**: Constant, Step Decay, Exponential, and Plateau schedules.
- **Numerical Stability**: Gradient clipping (by value and by Euclidean norm) and L1/L2 weight decay regularization.
- **Weight Initializers**: Xavier/Glorot (Sigmoid/Tanh), He/Kaiming (ReLU), and bounded Uniform.
- **Heterogeneous Architectures**: Arbitrary layer dimensions and inter-layer connectivity.
- **Structured Exceptions**: Typed exception system with dedicated numeric error codes and contextual data.
- **Model Persistence**: JSON serialization and deserialization.

---

## Installation

### Recommended: Autoloader

Include the built-in autoloader in your project:

```php
<?php
require_once __DIR__ . '/src/autoload.php';

use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;

$network = Network::Create('classifier', 2, 4);
$trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.01, new Adam());
```

---

## Quick Start

### 1. Creating a Network

```php
use mc\nina\Network;
use mc\nina\activation\Sigmoid;

// Creates a network with 2 layers of 4 neurons each
$network = Network::Create("MyNetwork", 2, 4, new Sigmoid());
```

### 2. Training with Backpropagation

```php
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;

$trainingData = [
    [[0.0, 0.0], [0.0]],
    [[0.0, 1.0], [1.0]],
    [[1.0, 0.0], [1.0]],
    [[1.0, 1.0], [0.0]],
];

$trainer = new BackpropagationTrainer(
    $network,
    new MeanSquaredError(),
    0.05,
    new Adam()
);

// Train for 200 epochs or until reaching 0.01 loss
$losses = $trainer->trainBatch($trainingData, epochs: 200, targetLoss: 0.01);
```

### 3. Training with Genetic Algorithm

```php
use mc\nina\Trainer;

// Population of 50 candidate networks
$trainer = new Trainer($network, pull: 50, mutationRate: 10);
$trainer->Train("input_str", "target_str");

$bestNetwork = $trainer->getBestNetwork();
```

### 4. Making Predictions

```php
use mc\nina\Predictor;

// Raw vector computation
$output = $network->Calculate([1.0, 0.0]);

// String/character prediction interface
$predictor = new Predictor($network);
$result = $predictor->Predict("input_data");
```

---

## Documentation

Comprehensive guides, specifications, and tutorials are located in the [`docs/`](docs/) directory:

| Guide | Description |
|---|---|
| **[Documentation Index](docs/README.md)** | Overview of all documentation resources |
| **[Backpropagation & Optimization](docs/backpropagation.md)** | Gradient descent, optimizers (Adam, SGD, RMSprop), schedules, and clipping |
| **[Genetic Algorithm](docs/genetic_algorithm.md)** | Evolutionary training, population mechanics, and mutations |
| **[API Reference](docs/api.md)** | Complete class, method, parameter, and return value reference |
| **[Architecture Guide](docs/architecture.md)** | System design, design patterns, and layer connectivity |
| **[Exception System](docs/exceptions.md)** | Error handling hierarchy and numeric error code reference |
| **[Use Cases & Advanced Usage](docs/use_cases.md)** | Practical tutorials: XOR, classification, function fitting, custom architectures |
| **[Quick Start Guide](QUICKSTART.md)** | Detailed walkthrough tutorial for new users |
| **[Russian Documentation](docs/ru/)** | Русскоязычная версия документации |

---

## Testing

Run the test suite via CLI:

```bash
php tests/test_cell.php
php tests/test_layer.php
php tests/test_exceptions.php
php tests/test_activation_derivatives.php
php tests/test_loss_functions.php
php tests/test_cell_backpropagation.php
php tests/test_backpropagation_trainer.php
php tests/test_optimizers.php
php tests/test_xor_backpropagation.php
php tests/test_performance.php
```

Or run all tests sequentially:

```bash
php -r "foreach (glob('tests/test_*.php') as \$test) { echo \"Running {\$test}...\n\"; passthru(\"php {\$test}\"); }"
```

---

## Project Structure

```txt
nina/
├── docs/                   # Full documentation suite
├── examples/               # Working demonstrations and sample models
├── src/                    # Library source code
│   ├── autoload.php        # Class autoloader and convenience aliases
│   └── mc/
│       ├── lib.php         # Numerical, UUID, and vector utility functions
│       └── nina/
│           ├── cell.php    # Neuron implementation with forward caching
│           ├── layer.php   # Layer container
│           ├── network.php # Neural network pipeline
│           ├── trainer.php # Genetic algorithm trainer
│           ├── predictor.php # Prediction interface
│           ├── BackpropagationTrainer.php # Gradient descent training engine
│           ├── activation/ # Sigmoid, Tanh, ReLU, Softmax
│           ├── exception/  # Typed exceptions
│           ├── initializer/# Xavier, He, RandomUniform initializers
│           ├── loss/       # MSE and Cross-Entropy loss functions
│           ├── optimizer/  # SGD, Adam, RMSprop optimizers
│           └── schedule/   # Learning rate decay schedules
└── tests/                  # Automated unit and integration test suites
```

---

## Contributing

Contributions are welcome. Please ensure that all existing unit tests pass before submitting pull requests.

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
