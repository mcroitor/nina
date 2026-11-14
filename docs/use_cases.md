# NiNa Neural Network Library - Use Cases & Advanced Usage

This guide covers practical use cases, advanced network construction, model persistence, and custom workflows.

---

## Table of Contents

- [Use Cases](#use-cases)
  - [1. Data Classification (Binary & Multi-class)](#1-data-classification)
  - [2. XOR Problem](#2-xor-problem)
  - [3. Function Approximation](#3-function-approximation)
  - [4. Pattern Recognition](#4-pattern-recognition)
- [Advanced Usage](#advanced-usage)
  - [Custom Layer-by-Layer Architecture](#custom-layer-by-layer-architecture)
  - [Model Serialization (Save & Load)](#model-serialization-save--load)
  - [Error Handling & Recovery](#error-handling--recovery)
  - [Custom Training Loops](#custom-training-loops)

---

## Use Cases

### 1. Data Classification

Binary classification models can distinguish between two categories using a Sigmoid output neuron:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\activation\Sigmoid;
use mc\nina\optimizer\Adam;

// 2 input features -> hidden layer (4 neurons) -> 1 binary output
$network = Network::Create("BinaryClassifier", 2, 4, new Sigmoid());

$trainingData = [
    [[0.1, 0.2], [0.0]],
    [[0.2, 0.1], [0.0]],
    [[0.8, 0.9], [1.0]],
    [[0.9, 0.8], [1.0]],
];

$trainer = new BackpropagationTrainer(
    $network,
    new MeanSquaredError(),
    0.05,
    new Adam()
);

$losses = $trainer->trainBatch($trainingData, 200, 0.01);

// Test prediction
$output = $network->Calculate([0.85, 0.95]);
$class = ($output[0] >= 0.5) ? 'Positive' : 'Negative';
echo "Predicted: {$class} (confidence: " . round($output[0], 3) . ")\n";
```

### 2. XOR Problem

The XOR problem is a classic non-linear benchmark requiring hidden layer representations:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Cell;
use mc\nina\Layer;
use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\activation\Sigmoid;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\initializer\XavierInitializer;

// Heterogeneous architecture: 2 inputs -> 4 hidden neurons -> 1 output neuron
$hiddenLayer = new Layer("hidden", [
    new Cell("h0", [0.5, -0.4], new Sigmoid()),
    new Cell("h1", [-0.3, 0.7], new Sigmoid()),
    new Cell("h2", [0.6, 0.8], new Sigmoid()),
    new Cell("h3", [-0.5, -0.6], new Sigmoid()),
]);

$outputLayer = new Layer("output", [
    new Cell("out0", [0.8, -0.5, 0.4, -0.7], new Sigmoid())
]);

$network = new Network("xor_net", [$hiddenLayer, $outputLayer]);

$xorData = [
    [[0.0, 0.0], [0.0]],
    [[0.0, 1.0], [1.0]],
    [[1.0, 0.0], [1.0]],
    [[1.0, 1.0], [0.0]],
];

$trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.1, new Adam());
$trainer->trainBatch($xorData, 500, 0.01);

foreach ($xorData as $example) {
    $pred = $network->Calculate($example[0])[0];
    echo "XOR(" . implode(',', $example[0]) . ") = " . round($pred) . " (raw: " . round($pred, 3) . ")\n";
}
```

### 3. Function Approximation

Neural networks can approximate continuous mathematical functions:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;
use mc\nina\Trainer;

// Approximate f(x) using Genetic Algorithm
$network = Network::Create("Approximator", 3, 8);
$trainer = new Trainer($network, 50);

for ($epoch = 0; $epoch < 500; $epoch++) {
    $x = mt_rand(-100, 100) / 10.0;
    $y = $x * $x; // Target: f(x) = x^2
    $trainer->Train((string)$x, (string)$y);
}

$bestNetwork = $trainer->getBestNetwork();
```

### 4. Pattern Recognition

Classifying character representations or sprite patterns:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;
use mc\nina\Trainer;

$patterns = [
    ['pattern_a', 'type_1'],
    ['pattern_b', 'type_2'],
    ['pattern_c', 'type_1'],
];

$network = Network::Create("PatternRecognizer", 2, 10);
$trainer = new Trainer($network, 50);

foreach ($patterns as $pattern) {
    $trainer->Train($pattern[0], $pattern[1]);
}
```

---

## Advanced Usage

### Custom Layer-by-Layer Architecture

You can construct networks with distinct neuron configurations, activation functions, and inter-layer sizes:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Cell;
use mc\nina\Layer;
use mc\nina\Network;
use mc\nina\activation\ReLU;
use mc\nina\activation\Sigmoid;

// Layer 1: 3 neurons, each with 3 inputs (ReLU activation)
$inputCells = [
    new Cell("input_1", [0.5, 0.3, -0.2], new ReLU()),
    new Cell("input_2", [0.1, 0.8, 0.4], new ReLU()),
    new Cell("input_3", [-0.3, 0.6, 0.2], new ReLU()),
];
$inputLayer = new Layer("InputLayer", $inputCells);

// Layer 2: 2 hidden neurons, each receiving 3 inputs from previous layer (Sigmoid)
$hiddenCells = [
    new Cell("hidden_1", [0.7, -0.1, 0.4], new Sigmoid()),
    new Cell("hidden_2", [0.2, 0.5, -0.6], new Sigmoid()),
];
$hiddenLayer = new Layer("HiddenLayer", $hiddenCells);

// Layer 3: 1 output neuron, receiving 2 inputs from hidden layer (Sigmoid)
$outputCells = [
    new Cell("output_1", [0.3, -0.4], new Sigmoid()),
];
$outputLayer = new Layer("OutputLayer", $outputCells);

// Assemble network
$network = new Network("DeepClassifier", [$inputLayer, $hiddenLayer, $outputLayer]);

$output = $network->Calculate([1.0, 0.5, -0.2]);
```

### Model Serialization (Save & Load)

Trained networks can be serialized to JSON and loaded across environments:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;

// Save network state
$json = $network->Json();
file_put_contents('model.json', $json);

// Restore network
$loadedJson = file_get_contents('model.json');
$restoredNetwork = Network::Load($loadedJson);

assert($restoredNetwork->GetNrLayers() === $network->GetNrLayers());
```

### Error Handling & Recovery

NiNa uses structured, typed exceptions with numeric error codes and contextual payloads:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;
use mc\nina\exception\ValidationException;
use mc\nina\exception\NetworkException;
use mc\nina\exception\ComputationException;

try {
    $network = Network::Create("", 0, -5);
} catch (ValidationException $e) {
    echo "Validation error: " . $e->getMessage() . "\n";
    echo "Error Code: " . $e->getCode() . "\n";
    echo "Context: " . json_encode($e->getContext()) . "\n";
} catch (NetworkException $e) {
    echo "Architecture error: " . $e->getMessage() . "\n";
} catch (ComputationException $e) {
    echo "Numerical error: " . $e->getDetailedMessage() . "\n";
}
```

### Custom Training Loops

When fine-grained control over batching or gradient accumulation is needed:

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;

$trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.01, new Adam());

for ($epoch = 0; $epoch < 100; $epoch++) {
    $totalLoss = 0.0;
    
    foreach ($dataset as $example) {
        $loss = $trainer->trainSingle($example['input'], $example['target']);
        $totalLoss += $loss;
    }
    
    $avgLoss = $totalLoss / count($dataset);
    
    if ($epoch % 10 === 0) {
        echo "Epoch {$epoch}: loss = " . round($avgLoss, 6) . "\n";
    }
}
```
