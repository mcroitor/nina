<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\NetworkException;
use mc\nina\exception\SerializationException;
use mc\nina\activation\ActivationFunction;

/**
 * Complete neural network composed of sequential layers.
 * 
 * The Network class represents a full neural network with multiple layers
 * connected sequentially. Data flows from input through hidden layers to output.
 * Supports forward propagation, genetic algorithm training, and serialization.
 * 
 * ## Technical Overview
 * 
 * ### Architecture Constraints
 * - All layers must have the same size (number of neurons)
 * - Minimum 1 layer required
 * - Sequential feed-forward topology only
 * - Uniform layer size ensures compatibility with genetic algorithms
 * 
 * ### Data Flow
 * 1. Input vector → Layer 1 → Layer 2 → ... → Layer N → Output vector
 * 2. Each layer processes the output of the previous layer
 * 3. Input size is automatically adjusted to match first layer size
 * 4. Output size equals the size of the last layer
 * 
 * ### Genetic Algorithm Support
 * - Networks can spawn offspring with mutated weights
 * - Mutation rate controlled by percentage parameter
 * - New offspring networks receive unique generated names
 * - Original network structure preserved during mutation
 * 
 * ### Serialization Format
 * ```json
 * {
 *   "name": "network_identifier", 
 *   "layers": [
 *     {
 *       "name": "layer_1",
 *       "cells": [
 *         {"name": "cell_1", "dna": [0.5, -0.3, 0.8]},
 *         {"name": "cell_2", "dna": [0.1, 0.7, -0.2]}
 *       ]
 *     }
 *   ]
 * }
 * ```
 * 
 * ### Performance Characteristics
 * - Memory: O(L × N × N) where L=layers, N=neurons per layer
 * - Forward pass: O(L × N²) computational complexity
 * - Suitable for small to medium networks (≤1000 neurons per layer)
 * 
 * ### Use Cases
 * - Pattern recognition and classification
 * - Function approximation
 * - Evolutionary neural network training
 * - Rapid prototyping of neural architectures
 * 
 * @package mc\nina
 * @author Mihail Croitor
 * @version 0.2.0
 * @since 0.1.0
 * 
 * @example Basic network creation and usage:
 * ```php
 * // Create a 3-layer network with 64 neurons each
 * $network = Network::Create("classifier", 3, 64);
 * 
 * // Process input data
 * $input = array_fill(0, 64, 0.5);
 * $output = $network->Calculate($input);
 * 
 * // Save to file
 * file_put_contents('model.json', $network->Json());
 * 
 * // Load from file
 * $loadedNetwork = Network::Load(file_get_contents('model.json'));
 * ```
 * 
 * @example Genetic algorithm usage:
 * ```php
 * $parent = Network::Create("parent", 2, 32);
 * $child = $parent->Born(10); // 10% mutation rate
 * 
 * // Child has same structure but mutated weights
 * // and different generated name
 * ```
 */
class Network {
    /**
     * Unique identifier for this network
     * @var string
     */
    private string $name;
    
    /**
     * Array of Layer objects that make up this network
     * @var array<Layer>
     */
    private array $layers;

    /**
     * Constructs a new neural network with specified layers.
     * 
     * Creates a neural network by combining multiple Layer objects into a sequential
     * feed-forward architecture. All layers must have the same size to ensure
     * compatibility with the genetic algorithm training system.
     * 
     * ## Validation Rules
     * - Network name must be non-empty after trimming whitespace
     * - At least one layer is required
     * - All layers must be valid Layer instances
     * - All layers must have identical size (number of neurons)
     * 
     * ## Architecture Constraints
     * The uniform layer size requirement ensures:
     * - Consistent data flow between layers
     * - Compatibility with genetic mutation operations
     * - Simplified serialization and deserialization
     * - Predictable memory allocation patterns
     * 
     * @param string $name Unique identifier for this network (must be non-empty)
     * @param array<Layer> $layers Array of Layer objects (must be non-empty, uniform size)
     * 
     * @throws ValidationException When name is empty or layers contain invalid types
     * @throws NetworkException When layers array is empty or sizes don't match
     * 
     * @example Creating a custom network:
     * ```php
     * $layer1 = new Layer("input", [$cell1, $cell2, $cell3]);
     * $layer2 = new Layer("hidden", [$cell4, $cell5, $cell6]);
     * $layer3 = new Layer("output", [$cell7, $cell8, $cell9]);
     * 
     * $network = new Network("my_classifier", [$layer1, $layer2, $layer3]);
     * ```
     */
    public function __construct(string $name, array $layers) {
        // Validate name
        if (empty(trim($name))) {
            throw ValidationException::invalidValue('name', $name, 'non-empty string');
        }
        
        // Validate layers array
        if (empty($layers)) {
            throw NetworkException::invalidArchitecture(
                'network must contain at least one layer',
                ['layers_count' => 0]
            );
        }
        
        // Validate each layer
        foreach ($layers as $index => $layer) {
            if (!($layer instanceof Layer)) {
                throw ValidationException::invalidValue(
                    "layers[{$index}]",
                    $layer,
                    'Layer instance',
                    "got " . gettype($layer)
                );
            }
        }
        
        // Validate layer size consistency
        $firstLayerSize = $layers[0]->GetSize();
        foreach ($layers as $index => $layer) {
            if ($layer->GetSize() !== $firstLayerSize) {
                throw NetworkException::invalidArchitecture(
                    "All layers must have the same size",
                    [
                        'layer_index' => $index,
                        'layer_size' => $layer->GetSize(),
                        'expected_size' => $firstLayerSize
                    ]
                );
            }
        }
        
        $this->name = trim($name);
        $this->layers = $layers;
    }

    /**
     * Factory method to create a uniform neural network.
     * 
     * Creates a new network with the specified number of layers, each containing
     * the same number of neurons. All neurons are initialized with random weights
     * and the specified activation function. This is the recommended way to create
     * networks for genetic algorithm training.
     * 
     * ## Network Structure
     * - All layers have identical size (layerSize neurons each)
     * - Each neuron has layerSize weights (fully connected)
     * - Random weight initialization in range [-1.0, 1.0]
     * - Uniform activation function across all neurons
     * 
     * ## Memory Requirements
     * Total weights: nrLayers × layerSize × layerSize
     * Example: 3 layers × 64 neurons = 12,288 weights
     * 
     * @param string $name Unique identifier for the network (must be non-empty)
     * @param int $nrLayers Number of layers to create (must be positive)
     * @param int $layerSize Number of neurons per layer (must be positive)
     * @param ActivationFunction|null $activationFunction Function for all neurons (null = ReLU default)
     * 
     * @return Network New network with random initialization
     * 
     * @throws ValidationException When name is empty
     * @throws NetworkException When nrLayers or layerSize is not positive
     * 
     * @example Creating standard architectures:
     * ```php
     * // Small network for quick prototyping (uses default ReLU)
     * $small = Network::Create("prototype", 2, 16);
     * 
     * // Medium network for pattern recognition (explicit Sigmoid)
     * $medium = Network::Create("classifier", 3, 64, new Sigmoid());
     * 
     * // Large network for complex tasks (explicit Tanh)
     * $large = Network::Create("deep_net", 5, 128, new Tanh());
     * ```
     */
    public static function Create(string $name, int $nrLayers, int $layerSize, ?ActivationFunction $activationFunction = null): Network {
        // Validate parameters
        if (empty(trim($name))) {
            throw ValidationException::invalidValue('name', $name, 'non-empty string');
        }
        
        if ($nrLayers <= 0) {
            throw NetworkException::configurationError('nrLayers', $nrLayers, 'must be positive');
        }
        
        if ($layerSize <= 0) {
            throw NetworkException::configurationError('layerSize', $layerSize, 'must be positive');
        }
        
        $layers = [];
        for ($i = 0; $i < $nrLayers; $i++) {
            $layers[] = Layer::Create($layerSize, $activationFunction);
        }
        return new Network($name, $layers);
    }

    /**
     * Gets the unique name identifier of this network.
     * 
     * Returns the name assigned during network construction or loading.
     * This identifier is used for debugging, logging, and model persistence.
     * 
     * @return string The network's name/identifier
     * 
     * @example
     * ```php
     * $network = Network::Create("classifier", 3, 64);
     * echo $network->GetName(); // Output: "classifier"
     * ```
     */
    public function GetName(): string {
        return $this->name;
    }

    /**
     * Gets the total number of layers in this network.
     * 
     * Returns the count of sequential layers that make up the network.
     * This includes all hidden layers plus input/output processing layers.
     * 
     * @return int Number of layers in the network (always ≥ 1)
     * 
     * @example
     * ```php
     * $network = Network::Create("deep", 5, 32);
     * echo $network->GetNrLayers(); // Output: 5
     * ```
     */
    public function GetNrLayers(): int {
        return count($this->layers);
    }

    /**
     * Gets the uniform size of all layers in this network.
     * 
     * Returns the number of neurons in each layer. Since all layers must have
     * the same size in NiNa architecture, this value applies to every layer.
     * 
     * @return int Number of neurons per layer (always ≥ 1)
     * 
     * @example
     * ```php
     * $network = Network::Create("classifier", 3, 64);
     * echo $network->GetLayerSize(); // Output: 64
     * 
     * // All layers have the same size
     * for ($i = 0; $i < $network->GetNrLayers(); $i++) {
     *     assert($network->GetLayer($i)->GetSize() === 64);
     * }
     * ```
     */
    public function GetLayerSize(): int {
        return $this->layers[0]->GetSize();
    }

    /**
     * Gets a specific layer from the network by index.
     * 
     * Provides access to individual Layer objects within the network.
     * Useful for inspecting layer properties, accessing specific neurons,
     * or implementing custom training algorithms.
     * 
     * @param int $index Zero-based index of the layer to retrieve
     * 
     * @return Layer The layer at the specified index
     * 
     * @throws ValidationException When index is out of bounds
     * 
     * @example Accessing network layers:
     * ```php
     * $network = Network::Create("net", 3, 16);
     * 
     * $inputLayer = $network->GetLayer(0);    // First layer
     * $hiddenLayer = $network->GetLayer(1);   // Middle layer  
     * $outputLayer = $network->GetLayer(2);   // Last layer
     * 
     * // Access specific neurons
     * $firstNeuron = $inputLayer->GetCell(0);
     * ```
     */
    public function GetLayer(int $index): Layer {
        if ($index < 0 || $index >= count($this->layers)) {
            throw ValidationException::outOfRange(
                'layer_index',
                $index,
                0,
                count($this->layers) - 1
            );
        }
        return $this->layers[$index];
    }

    /**
     * Creates a mutated offspring network for genetic algorithm training.
     * 
     * Generates a new network with the same architecture but with mutated weights
     * in all neurons. This is the core operation for evolutionary neural network
     * training, where populations of networks evolve through mutation and selection.
     * 
     * ## Mutation Process
     * 1. Each layer creates a mutated copy of itself
     * 2. Each neuron's weights are randomly adjusted
     * 3. Mutation magnitude controlled by percentage parameter
     * 4. Network structure (layers, sizes) remains unchanged
     * 5. New network receives a unique generated name
     * 
     * ## Genetic Algorithm Usage
     * - Create population by spawning multiple offspring
     * - Evaluate fitness of each network variant
     * - Select best performers as parents for next generation
     * - Repeat until satisfactory performance achieved
     * 
     * @param int $percent Mutation percentage (0-100) controlling weight change magnitude
     * 
     * @return Network New network with mutated weights and unique name
     * 
     * @example Basic genetic algorithm pattern:
     * ```php
     * $parent = Network::Create("parent", 3, 32);
     * 
     * // Create offspring population
     * $offspring = [];
     * for ($i = 0; $i < 50; $i++) {
     *     $offspring[] = $parent->Born(10); // 10% mutation rate
     * }
     * 
     * // Each offspring has same structure but different weights
     * // Evaluate fitness and select best for next generation
     * ```
     * 
     * @example Mutation rate effects:
     * ```php
     * $conservative = $parent->Born(1);   // Small changes (fine-tuning)
     * $moderate = $parent->Born(10);      // Balanced exploration/exploitation  
     * $aggressive = $parent->Born(50);    // Large changes (broad exploration)
     * ```
     */
    public function Born(int $percent): Network {
        $layers = [];
        foreach ($this->layers as $layer) {
            $layers[] = $layer->Born($percent);
        }
        return new Network(\mc\uuid(), $layers);
    }

    /**
     * Performs forward propagation through the entire network.
     * 
     * Processes input data through all layers sequentially to produce final output.
     * This is the core inference operation used for predictions, classifications,
     * and fitness evaluation during training.
     * 
     * ## Processing Pipeline
     * 1. Input size validation and automatic adjustment
     * 2. Sequential processing through each layer
     * 3. Each layer output becomes input for next layer
     * 4. Final layer output returned as network result
     * 
     * ## Input Size Handling
     * - **Too small**: Padded with zeros to match layer size
     * - **Too large**: Truncated to match layer size  
     * - **Exact match**: Processed as-is
     * 
     * ## Data Flow Example
     * ```
     * Input[64] → Layer1[64] → Hidden[64] → Layer3[64] → Output[64]
     * ```
     * 
     * ## Performance Notes
     * - Computational complexity: O(L × N²) where L=layers, N=neurons
     * - Memory usage: O(N) for intermediate results
     * - No gradient computation (forward-only)
     * 
     * @param array<float> $values Input values to process (automatically resized)
     * 
     * @return array<float> Final network outputs (size equals layer size)
     * 
     * @throws ValidationException When input array is empty
     * @throws NetworkException When layer calculation fails
     * 
     * @example Basic usage:
     * ```php
     * $network = Network::Create("classifier", 3, 64);
     * 
     * // Classification example
     * $imagePixels = array_fill(0, 64, 0.5); // 64 pixel values
     * $classification = $network->Calculate($imagePixels);
     * 
     * // Result interpretation
     * $confidence = array_sum($classification) / count($classification);
     * $isPositive = $confidence > 0.5;
     * ```
     * 
     * @example Size adjustment behavior:
     * ```php
     * $network = Network::Create("net", 2, 10);
     * 
     * // Small input (padded with zeros)
     * $small = [1, 2, 3]; // → [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
     * $result1 = $network->Calculate($small);
     * 
     * // Large input (truncated)  
     * $large = range(1, 20); // → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
     * $result2 = $network->Calculate($large);
     * 
     * // Perfect size (no adjustment)
     * $perfect = range(1, 10); // → [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
     * $result3 = $network->Calculate($perfect);
     * ```
     */
    public function Calculate(array $values): array {
        if (empty($values)) {
            throw ValidationException::invalidValue('input_values', 'empty array', 'non-empty array');
        }
        
        $expectedSize = $this->GetLayerSize();
        
        // Adjust input size to match network
        if(count($values) < $expectedSize) {
            $values = array_pad($values, $expectedSize, 0);
        }
        if(count($values) > $expectedSize) {
            $values = array_slice($values, 0, $expectedSize);
        }
        
        $result = $values;
        foreach ($this->layers as $layerIndex => $layer) {
            try {
                $result = $layer->Calculate($result);
            } catch (\Exception $e) {
                throw NetworkException::incompatibleOperation(
                    'layer calculation',
                    "Failed to calculate layer {$layerIndex} in network '{$this->name}': " . $e->getMessage()
                );
            }
        }
        return $result;
    }

    /**
     * Prints comprehensive network information to console for debugging.
     * 
     * Outputs a detailed hierarchical view of the entire network structure,
     * including all layers and their neurons. This is invaluable for debugging,
     * development, and understanding network architecture.
     * 
     * ## Output Format
     * ```
     * Network: network_name
     * Layer: layer_1_name
     *   cell_1_name (ActivationFunction): weight1 weight2 weight3 ...
     *   cell_2_name (ActivationFunction): weight1 weight2 weight3 ...
     * Layer: layer_2_name  
     *   cell_3_name (ActivationFunction): weight1 weight2 weight3 ...
     *   ...
     * ```
     * 
     * ## Use Cases
     * - Debugging network initialization
     * - Verifying network structure after loading
     * - Monitoring weight evolution during training
     * - Educational visualization of network architecture
     * - Troubleshooting serialization issues
     * 
     * @return void Outputs directly to console/stdout
     * 
     * @example Development debugging:
     * ```php
     * $network = Network::Create("debug_net", 2, 3);
     * $network->Print();
     * 
     * // Output might look like:
     * // Network: debug_net
     * // Layer: uuid-layer-1
     * //   uuid-cell-1 (ReLU): 0.123 -0.456 0.789
     * //   uuid-cell-2 (ReLU): -0.234 0.567 -0.890
     * //   uuid-cell-3 (ReLU): 0.345 -0.678 0.901
     * // Layer: uuid-layer-2
     * //   uuid-cell-4 (ReLU): 0.234 -0.567 0.890
     * //   ...
     * ```
     * 
     * @example Training monitoring:
     * ```php
     * echo "Before training:\n";
     * $network->Print();
     * 
     * // ... training process ...
     * 
     * echo "\nAfter training:\n";
     * $trainedNetwork->Print();
     * ```
     */
    public function Print(): void {
        echo "Network: {$this->name}" . PHP_EOL;
        foreach ($this->layers as $layer) {
            $layer->Print();
        }
    }

    /**
     * Creates a Network instance from JSON serialization data.
     * 
     * Reconstructs a complete neural network from its JSON representation,
     * restoring all layers, neurons, weights, and activation functions.
     * This enables model persistence, sharing, and deployment across different
     * environments and applications.
     * 
     * ## JSON Structure Requirements
     * - Root object with "name" and "layers" fields
     * - Each layer contains "name" and "cells" arrays
     * - Each cell has "name" and "dna" (weights) fields
     * - Activation functions restored based on layer configuration
     * 
     * ## Validation Process
     * 1. JSON syntax and structure validation
     * 2. Required field presence verification
     * 3. Individual layer reconstruction and validation
     * 4. Network architecture consistency checks
     * 5. Complete network assembly
     * 
     * ## Error Recovery
     * - Detailed error messages for debugging
     * - Specific layer/cell error identification
     * - Preserves original JSON structure information
     * - Fails fast on first corruption detected
     * 
     * @param string $json JSON string containing network data (must be non-empty)
     * 
     * @return Network Fully reconstructed network instance
     * 
     * @throws SerializationException When JSON is invalid or structure is malformed
     * @throws ValidationException When loaded data fails validation
     * @throws NetworkException When network assembly fails
     * 
     * @example Basic model loading:
     * ```php
     * // Load from file
     * $json = file_get_contents('trained_model.json');
     * $network = Network::Load($json);
     * 
     * // Verify loaded network
     * echo "Loaded: " . $network->GetName();
     * echo "Layers: " . $network->GetNrLayers();
     * echo "Size: " . $network->GetLayerSize();
     * ```
     * 
     * @example Error handling:
     * ```php
     * try {
     *     $network = Network::Load($suspiciousJson);
     *     echo "Network loaded successfully";
     * } catch (SerializationException $e) {
     *     echo "JSON corruption: " . $e->getMessage();
     * } catch (ValidationException $e) {
     *     echo "Invalid data: " . $e->getMessage();
     * } catch (NetworkException $e) {
     *     echo "Architecture issue: " . $e->getMessage();
     * }
     * ```
     * 
     * @example Model deployment:
     * ```php
     * // Production model loading with validation
     * function loadProductionModel(string $modelPath): Network {
     *     if (!file_exists($modelPath)) {
     *         throw new RuntimeException("Model file not found");
     *     }
     *     
     *     $json = file_get_contents($modelPath);
     *     $network = Network::Load($json);
     *     
     *     // Validate expected architecture
     *     if ($network->GetLayerSize() !== 64) {
     *         throw new RuntimeException("Unexpected model architecture");
     *     }
     *     
     *     return $network;
     * }
     * ```
     */
    public static function Load(string $json): Network {
        if (empty(trim($json))) {
            throw SerializationException::invalidFormat('JSON string', 'empty string');
        }
        
        $data = json_decode($json, true);
        
        if ($data === null) {
            throw SerializationException::jsonError('decode', json_last_error());
        }
        
        if (!isset($data['name']) || !isset($data['layers'])) {
            throw SerializationException::invalidFormat(
                'Network JSON with name and layers fields',
                'JSON missing required fields'
            );
        }
        
        $layers = [];
        foreach ($data["layers"] as $index => $layerData) {
            try {
                $layers[] = Layer::Load(json_encode($layerData));
            } catch (\Exception $e) {
                throw SerializationException::corruptedData(
                    'layer data',
                    "Failed to load layer at index {$index}: " . $e->getMessage()
                );
            }
        }
        
        return new Network($data["name"], $layers);
    }

    /**
     * Serializes the complete network to JSON format for persistence.
     * 
     * Converts the entire network state (architecture, weights, activation functions)
     * into a JSON string suitable for file storage, network transmission, or backup.
     * The resulting JSON can be used with Network::Load() to reconstruct the network.
     * 
     * ## Serialization Process
     * 1. Network metadata (name) extraction
     * 2. Sequential layer serialization
     * 3. Individual cell weight/configuration serialization  
     * 4. JSON structure assembly and encoding
     * 5. Integrity validation of final JSON
     * 
     * ## JSON Structure Generated
     * ```json
     * {
     *   "name": "network_identifier",
     *   "layers": [
     *     {
     *       "name": "layer_1",
     *       "cells": [
     *         {
     *           "name": "cell_1", 
     *           "dna": [0.123, -0.456, 0.789]
     *         },
     *         {
     *           "name": "cell_2",
     *           "dna": [-0.234, 0.567, -0.890]
     *         }
     *       ]
     *     }
     *   ]
     * }
     * ```
     * 
     * ## Use Cases
     * - Model persistence between sessions
     * - Trained network backup and versioning
     * - Model sharing and distribution
     * - Cross-platform deployment
     * - Checkpoint saving during long training
     * 
     * @return string JSON representation of the complete network
     * 
     * @throws SerializationException When JSON encoding fails
     * 
     * @example Model saving workflow:
     * ```php
     * // Train and save model
     * $network = Network::Create("classifier", 3, 64);
     * // ... training process ...
     * 
     * // Save to file
     * $json = $network->Json();
     * file_put_contents('models/classifier_v1.json', $json);
     * 
     * // Later: load and use
     * $loadedNetwork = Network::Load(file_get_contents('models/classifier_v1.json'));
     * $result = $loadedNetwork->Calculate($input);
     * ```
     * 
     * @example Model versioning:
     * ```php
     * // Save training checkpoints
     * for ($epoch = 0; $epoch < 100; $epoch++) {
     *     // ... training iteration ...
     *     
     *     if ($epoch % 10 === 0) {
     *         $checkpoint = $bestNetwork->Json();
     *         file_put_contents("checkpoints/epoch_{$epoch}.json", $checkpoint);
     *     }
     * }
     * ```
     * 
     * @example Model comparison:
     * ```php
     * // Compare two network states
     * $before = $network->Json();
     * // ... mutations or training ...
     * $after = $network->Json();
     * 
     * if ($before !== $after) {
     *     echo "Network has changed";
     * }
     * ```
     */
    public function Json(): string {
        $data = [
            "name" => $this->name,
            "layers" => []
        ];
        
        foreach ($this->layers as $layer) {
            $layerJson = $layer->Json();
            $layerData = json_decode($layerJson, true);
            
            if ($layerData === null) {
                throw SerializationException::jsonError('decode', json_last_error());
            }
            
            $data["layers"][] = $layerData;
        }
        
        $json = json_encode($data);
        
        if ($json === false) {
            throw SerializationException::jsonError('encode', json_last_error());
        }
        
        return $json;
    }
}