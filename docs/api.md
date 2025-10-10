# API Documentation - NiNa Neural Network Library

## 📚 Table of Contents

- [Core Classes](#core-classes)
  - [Cell](#cell---neuron)
  - [Layer](#layer---layer)
  - [Network](#network---neural-network)
  - [Trainer](#trainer---trainer)
  - [Predictor](#predictor---predictor)
- [Activation Functions](#activation-functions)
- [Exception System](#exception-system)
- [Utilities](#utilities)
- [Usage Examples](#usage-examples)

---

## Core Classes

### Cell - Neuron

Basic unit of neural network - individual neuron with weights and activation function.

#### Constructors

```php
public function __construct(
    string $name, 
    array $dna, 
    ?ActivationFunction $activationFunction = null
)
```

**Parameters:**

- `$name` (string) - Unique neuron name
- `$dna` (array) - Array of weights (float values)
- `$activationFunction` (ActivationFunction|null) - Activation function (default ReLU)

**Exceptions:**

- `ValidationException` - On invalid parameters
- `ComputationException` - On incorrect numerical values

**Example:**

```php
use mc\nina\Cell;
use mc\nina\activation\Sigmoid;

$cell = new Cell("neuron_1", [0.5, -0.3, 0.8], new Sigmoid());
```

```php
public static function Create(
    string $name, 
    int $size, 
    float $value = 0.0, 
    ?ActivationFunction $activationFunction = null
): Cell
```

**Parameters:**

- `$name` (string) - Neuron name
- `$size` (int) - Number of weights (input layer size)
- `$value` (float) - Initial value for all weights
- `$activationFunction` (ActivationFunction|null) - Activation function

**Returns:** `Cell` - New neuron object

**Exceptions:**

- `ValidationException` - When size <= 0 or invalid name
- `ComputationException` - On incorrect value

**Example:**

```php
$cell = Cell::Create("neuron_1", 5, 0.1, new ReLU());
```

#### Main Methods

```php
public function Calculate(array $values): float
```

Calculates neuron output (forward propagation with activation).

**Parameters:**

- `$values` (array) - Input values for the neuron

**Returns:** `float` - Activated neuron output

**Exceptions:**

- `ValidationException` - On input/weight size mismatch
- `ComputationException` - On numerical errors or invalid values

**Example:**

```php
$input = [1.0, 0.5, -0.3];
$output = $cell->Calculate($input);
echo "Output: {$output}"; // Output: 0.73105
```

```php
public function CalculateRaw(array $values): float
```

Calculates weighted sum without applying activation function.

**Parameters:**

- `$values` (array) - Input values

**Returns:** `float` - Raw output (before activation)

**Example:**

```php
$rawOutput = $cell->CalculateRaw($input);
echo "Raw output: {$rawOutput}"; // Raw output: 1.1
```

```php
public function Born(int $percent): Cell
```

Creates new neuron based on current with weight mutation.

**Parameters:**

- `$percent` (int) - Mutation percentage (0-100)

**Returns:** `Cell` - New mutated neuron

**Example:**

```php
$mutatedCell = $cell->Born(10); // 10% mutation
```

#### Access Methods

```php
public function GetName(): string
```

Returns neuron name.

```php
public function GetDna(): array
```

Returns neuron weights array.

```php
public function GetSize(): int
```

Returns number of weights (input size).

```php
public function GetValue(int $index): float
```

Returns specific weight by index.

```php
public function GetActivationFunction(): ActivationFunction
```

Returns neuron activation function.

```php
public function SetActivationFunction(ActivationFunction $activationFunction): void
```

Sets new activation function.

#### Serialization

```php
public function Json(): string
```

Serializes neuron to JSON format.

**Returns:** `string` - JSON representation of neuron

**Exceptions:**

- `SerializationException` - On JSON encoding errors

```php
public static function Load(string $json): Cell
```

Loads neuron from JSON.

**Parameters:**

- `$json` (string) - JSON string

**Returns:** `Cell` - Restored neuron

**Exceptions:**

- `SerializationException` - On JSON parsing errors

---

### Layer - Layer

Collection of neurons working in parallel.

#### Constructors

```php
public function __construct(string $name, array $cells)
```

**Parameters:**

- `$name` (string) - Layer name
- `$cells` (array) - Array of Cell objects

**Exceptions:**

- `ValidationException` - On invalid name or incorrect cell types
- `LayerException` - On empty cell array

```php
public static function Create(int $size): Layer
```

**Parameters:**

- `$size` (int) - Number of neurons in layer

**Returns:** `Layer` - New layer

**Exceptions:**

- `LayerException` - When size <= 0

#### Main Methods

```php
public function Calculate(array $values): array
```

Processes input data through all layer neurons.

**Parameters:**

- `$values` (array) - Input data

**Returns:** `array` - Array of outputs from all neurons

**Exceptions:**

- `ValidationException` - On empty input data
- `LayerException` - On neuron calculation errors

```php
public function Born(int $percent): Layer
```

Creates new layer with mutated neurons.

**Parameters:**

- `$percent` (int) - Mutation percentage

**Returns:** `Layer` - New mutated layer

#### Access Methods

```php
public function GetName(): string
```

Returns layer name.

```php
public function GetCell(int $index): Cell
```

Returns neuron by index.

**Exceptions:**

- `ValidationException` - On invalid index

```php
public function GetSize(): int
```

Returns number of neurons in layer.

#### Serialization

```php
public function Json(): string
public static function Load(string $json): Layer
```

Similar to Cell methods with array handling.

---

### Network - Neural Network

Complete neural network of sequentially connected layers.

#### Constructors

```php
public function __construct(string $name, array $layers)
```

**Parameters:**

- `$name` (string) - Network name
- `$layers` (array) - Array of Layer objects

**Exceptions:**

- `ValidationException` - On invalid name or layer types
- `NetworkException` - On empty layer array or incompatible sizes

```php
public static function Create(string $name, int $nrLayers, int $layerSize): Network
```

**Parameters:**

- `$name` (string) - Network name
- `$nrLayers` (int) - Number of layers
- `$layerSize` (int) - Size of each layer

**Returns:** `Network` - New network

**Exceptions:**

- `ValidationException` - On invalid name
- `NetworkException` - On non-positive parameter values

#### Main Methods

```php
public function Calculate(array $values): array
```

Performs forward propagation through entire network.

**Parameters:**

- `$values` (array) - Input data

**Returns:** `array` - Final network outputs

**Features:**

- Automatically adjusts input data size to first layer size
- If input data is smaller - pads with zeros
- If larger - truncates to required size

**Exceptions:**

- `ValidationException` - On empty input data
- `NetworkException` - On layer calculation errors

```php
public function Born(int $percent): Network
```

Creates new network with mutated layers.

**Parameters:**

- `$percent` (int) - Mutation percentage

**Returns:** `Network` - New mutated network

#### Access Methods

```php
public function GetName(): string
```

Returns network name.

```php
public function GetNrLayers(): int
```

Returns number of layers.

```php
public function GetLayerSize(): int
```

Returns layer size (all layers have same size).

```php
public function GetLayer(int $index): Layer
```

Returns layer by index.

**Exceptions:**

- `ValidationException` - On invalid index

#### Serialization

```php
public function Json(): string
public static function Load(string $json): Network
```

Complete serialization/deserialization of network structure.

---

### Trainer - Trainer

Implements genetic algorithm for neural network training.

#### Constructor

```php
public function __construct(Network $network, int $pull)
```

**Parameters:**

- `$network` (Network) - Base network for evolution
- `$pull` (int) - Population size (1-1000)

**Exceptions:**

- `ValidationException` - On incorrect network type
- `TrainingException` - On invalid population size

#### Main Methods

```php
public function Train(string $input, string $expected): void
```

Performs one training cycle with given data.

**Parameters:**

- `$input` (string) - Training input data
- `$expected` (string) - Expected result

**Algorithm:**

1. Tests all networks in population
2. Selects best by Manhattan distance
3. Creates new population based on best network

**Exceptions:**

- `TrainingException` - On empty data or calculation errors

```php
public function getBestNetwork(): Network
```

Returns best network from population.

```php
public function getNetworks(): array
```

Returns entire current population.

```php
public function getNetworkByIndex(int $index): Network
```

Returns network by index from population.

**Exceptions:**

- `ValidationException` - On invalid index

---

### Predictor - Predictor

Interface for getting predictions from trained network.

#### Constructor

```php
public function __construct(Network $network)
```

**Parameters:**

- `$network` (Network) - Trained network

**Exceptions:**

- `ValidationException` - On incorrect network type

#### Main Methods

```php
public function Predict(string $input): string
```

Gets prediction from network.

**Parameters:**

- `$input` (string) - Input data

**Returns:** `string` - Network prediction

**Process:**

1. Converts string to number array
2. Runs through network
3. Converts result back to string

**Exceptions:**

- `ValidationException` - On empty input data
- `ComputationException` - On calculation errors or incorrect results

---

## Activation Functions

### ActivationFunction Interface

```php
interface ActivationFunction
{
    public function activate(float $input): float;
    public function derivative(float $input): float;
    public function getName(): string;
}
```

### Available Functions

#### ReLU (Rectified Linear Unit)

```php
use mc\nina\activation\ReLU;

$relu = new ReLU();
$output = $relu->activate(-0.5); // 0.0
$output = $relu->activate(0.5);  // 0.5
```

**Characteristics:**

- Range: [0, ∞)
- Derivative: 1 if x > 0, else 0
- Application: Hidden layers, fast training

#### Sigmoid

```php
use mc\nina\activation\Sigmoid;

$sigmoid = new Sigmoid();
$output = $sigmoid->activate(0.0); // 0.5
$output = $sigmoid->activate(1.0); // 0.73105
```

**Characteristics:**

- Range: (0, 1)
- Derivative: σ(x)(1-σ(x))
- Application: Binary classification, output layers

#### Tanh (Hyperbolic Tangent)

```php
use mc\nina\activation\Tanh;

$tanh = new Tanh();
$output = $tanh->activate(0.0);  // 0.0
$output = $tanh->activate(1.0);  // 0.76159
```

**Characteristics:**

- Range: (-1, 1)
- Derivative: 1-tanh²(x)
- Application: Hidden layers, centered output

#### Softmax

```php
use mc\nina\activation\Softmax;

$softmax = new Softmax();
$output = $softmax->activate(1.0); // Context dependent
```

**Characteristics:**

- Range: (0, 1)
- Application: Multi-class classification
- Feature: Requires normalization across entire layer

---

## Exception System

### Base Class NinaException

```php
abstract class NinaException extends Exception
{
    public function getContext(): array;
    public function getSeverity(): string;
    public function getDetailedMessage(): string;
    public function toArray(): array;
}
```

### Exception Types

#### ValidationException (6001-6099)

```php
ValidationException::invalidValue(string $parameter, $value, string $expectedType, string $reason)
ValidationException::outOfRange(string $parameter, $value, $min, $max)
ValidationException::missingParameter(string $parameter, string $context)
ValidationException::invalidArraySize(string $parameter, int $actual, int $expected)
```

#### NetworkException (1001-1099)

```php
NetworkException::invalidArchitecture(string $reason, array $context)
NetworkException::configurationError(string $parameter, $value, string $reason)
NetworkException::incompatibleOperation(string $operation, string $reason)
```

#### LayerException (2001-2099)

```php
LayerException::invalidSize(int $size, string $reason)
LayerException::dimensionMismatch(int $expected, int $actual, string $context)
LayerException::invalidConnection(string $fromLayer, string $toLayer, string $reason)
LayerException::operationError(string $operation, string $reason)
```

#### TrainingException (3001-3099)

```php
TrainingException::invalidTrainingData(string $reason, array $context)
TrainingException::convergenceError(int $epoch, float $error, string $reason)
TrainingException::invalidParameters(string $parameter, $value, string $reason)
TrainingException::dataSizeMismatch(int $inputSize, int $expectedSize, string $dataType)
TrainingException::trainingTimeout(int $maxEpochs, float $timeElapsed)
```

#### SerializationException (4001-4099)

```php
SerializationException::fileError(string $operation, string $filename, string $reason)
SerializationException::invalidFormat(string $expectedFormat, string $actualFormat, string $context)
SerializationException::versionMismatch(string $modelVersion, string $libraryVersion)
SerializationException::corruptedData(string $dataType, string $reason)
SerializationException::jsonError(string $operation, int $jsonError)
```

#### ComputationException (5001-5099)

```php
ComputationException::domainError(string $operation, $value, string $reason)
ComputationException::divisionByZero(string $context)
ComputationException::matrixError(string $operation, array $dimensions, string $reason)
ComputationException::numericalOverflow(string $operation, $value, string $type)
ComputationException::invalidNumericalValue($value, string $context)
```

---

## Utilities

### lib.php Functions

```php
// Array creation
function array_create(int $size, float $value = 0.0): array

// Data conversion
function string2array(string $input): array
function array2string(array $input): string

// Genetic operations
function mutate(float $value, int $percent): float
function uuid(): string

// Mathematical functions
function manhattan_distance(array $a, array $b): float
```

---

## Usage Examples

### Basic Creation and Training

```php
<?php
use mc\nina\{Network, Trainer, Predictor};
use mc\nina\activation\ReLU;

// Create network
$network = Network::Create("MyNetwork", 3, 5);

// Create trainer
$trainer = new Trainer($network, 50);

// Training
$trainer->Train("input_data", "expected_output");

// Get best network
$bestNetwork = $trainer->getBestNetwork();

// Create predictor
$predictor = new Predictor($bestNetwork);

// Prediction
$result = $predictor->Predict("test_input");
echo "Prediction: {$result}";
```

### Custom Architecture

```php
<?php
use mc\nina\{Cell, Layer, Network};
use mc\nina\activation\{ReLU, Sigmoid};

// Create custom neurons
$cells = [
    new Cell("c1", [0.5, -0.3, 0.8], new ReLU()),
    new Cell("c2", [0.1, 0.7, -0.2], new Sigmoid())
];

// Create layer
$layer = new Layer("CustomLayer", $cells);

// Create network
$network = new Network("CustomNetwork", [$layer]);
```

### Error Handling

```php
<?php
use mc\nina\exception\{ValidationException, NetworkException, NinaException};

try {
    $network = Network::Create("", -1, 0); // Invalid parameters
} catch (ValidationException $e) {
    echo "Validation error: " . $e->getMessage();
    echo "Error code: " . $e->getCode();
    print_r($e->getContext());
} catch (NetworkException $e) {
    echo "Network error: " . $e->getDetailedMessage();
} catch (NinaException $e) {
    echo "General library error: " . $e->getMessage();
    
    // Debug logging
    error_log(json_encode($e->toArray()));
}
```

### Model Serialization

```php
<?php
// Save model
$networkJson = $network->Json();
file_put_contents('model.json', $networkJson);

// Load model
try {
    $savedJson = file_get_contents('model.json');
    $loadedNetwork = Network::Load($savedJson);
} catch (SerializationException $e) {
    echo "Failed to load model: " . $e->getMessage();
}
```

### Advanced Training

```php
<?php
function trainXORNetwork(): Network {
    $network = Network::Create("XOR", 3, 4);
    $trainer = new Trainer($network, 100);
    
    $trainingData = [
        ['00', '0'], ['01', '1'], 
        ['10', '1'], ['11', '0']
    ];
    
    // Multiple training epochs
    for ($epoch = 0; $epoch < 1000; $epoch++) {
        foreach ($trainingData as [$input, $expected]) {
            try {
                $trainer->Train($input, $expected);
            } catch (TrainingException $e) {
                echo "Training failed at epoch {$epoch}: " . $e->getMessage();
                break 2;
            }
        }
    }
    
    return $trainer->getBestNetwork();
}

$xorNetwork = trainXORNetwork();
```

---

## 📊 Return Codes and Status

### HTTP-like Error Codes

- **1xxx**: Network errors (NetworkException)
- **2xxx**: Layer errors (LayerException)  
- **3xxx**: Training errors (TrainingException)
- **4xxx**: Serialization errors (SerializationException)
- **5xxx**: Computation errors (ComputationException)
- **6xxx**: Validation errors (ValidationException)

### Severity Levels

- **error**: Critical errors requiring intervention
- **warning**: Warnings, work can continue
- **critical**: Catastrophic errors requiring shutdown

---

## 🔗 Additional Resources

- [Architecture Documentation](architecture.md)
- [Exception Guide](exceptions.md)
- [Examples and Tutorials](../examples/)
- [Tests](../tests/)

---

*Documentation is current for NiNa v0.2.0+*
