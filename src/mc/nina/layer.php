<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\LayerException;
use mc\nina\exception\SerializationException;

/**
 * Neural network layer containing multiple cells (neurons) working in parallel.
 * 
 * The Layer class represents a collection of neural cells that process inputs
 * simultaneously. Each cell in the layer receives the same input array and
 * produces its own output, creating a vector of outputs from the layer.
 * This is fundamental to neural network architecture where layers are stacked.
 * 
 * @package mc\nina
 * @author Mihail Croitor
 * @version 0.2.0
 * @since 0.1.0
 */
class Layer {
    /**
     * Unique identifier for this layer
     * @var string
     */
    private string $name;
    
    /**
     * Array of Cell objects that make up this layer
     * @var array<Cell>
     */
    private array $cells;

    /**
     * Constructs a new neural layer with specified cells.
     * 
     * Creates a layer containing multiple neural cells that will process
     * inputs in parallel. All cells must be valid Cell instances, and
     * the layer must contain at least one cell.
     * 
     * @param string $name Unique identifier for this layer (must be non-empty)
     * @param array<Cell> $cells Array of Cell objects (must be non-empty)
     * 
     * @throws ValidationException When name is empty or cells contain invalid types
     * @throws LayerException When cells array is empty
     * 
     * @example
     * ```php
     * $cells = [
     *     new Cell("cell1", [0.5, -0.3]),
     *     new Cell("cell2", [0.1, 0.8])
     * ];
     * $layer = new Layer("hidden_layer", $cells);
     * ```
     */
    public function __construct(string $name, array $cells) {
        // Validate name
        if (empty(trim($name))) {
            throw ValidationException::invalidValue('name', $name, 'non-empty string');
        }
        
        // Validate cells array
        if (empty($cells)) {
            throw LayerException::invalidSize(0, 'layer must contain at least one cell');
        }
        
        // Validate each cell
        foreach ($cells as $index => $cell) {
            if (!($cell instanceof Cell)) {
                throw ValidationException::invalidValue(
                    "cells[{$index}]",
                    $cell,
                    'Cell instance',
                    "got " . gettype($cell)
                );
            }
        }
        
        $this->name = trim($name);
        $this->cells = $cells;
    }

    /**
     * Gets the unique name identifier of this layer.
     * 
     * @return string The layer's name/identifier
     */
    public function GetName(): string {
        return $this->name;
    }

    /**
     * Gets a specific cell from this layer by index.
     * 
     * Retrieves the cell at the specified index from the layer's cell array.
     * Used for accessing individual neurons within the layer.
     * 
     * @param int $index Index of the cell to retrieve (0-based)
     * 
     * @return Cell The cell at the specified index
     * 
     * @throws ValidationException When index is out of bounds
     */
    public function GetCell(int $index): Cell {
        if ($index < 0 || $index >= count($this->cells)) {
            throw ValidationException::outOfRange(
                'index',
                $index,
                0,
                count($this->cells) - 1
            );
        }
        return $this->cells[$index];
    }

    /**
     * Gets the number of cells (neurons) in this layer.
     * 
     * @return int Number of cells in the layer
     */
    public function GetSize(): int {
        return count($this->cells);
    }

    /**
     * Creates a mutated offspring layer based on current layer (genetic algorithm operation).
     * 
     * Generates a new layer where each cell is a mutated version of the corresponding
     * cell in the current layer. This is essential for genetic algorithm training
     * where layers evolve through mutation and selection.
     * 
     * @param int $percent Mutation percentage (0-100) determining how much weights can change
     * 
     * @return Layer New layer with mutated cells
     * 
     * @example
     * ```php
     * $parentLayer = new Layer("layer1", $cells);
     * $childLayer = $parentLayer->Born(10); // 10% mutation rate
     * ```
     */
    public function Born(int $percent): Layer {
        $cells = [];
        foreach ($this->cells as $cell) {
            $cells[] = $cell->Born($percent);
        }
        return new Layer($this->name, $cells);
    }

    /**
     * Processes input data through all cells in the layer in parallel.
     * 
     * Each cell in the layer receives the same input array and produces its own output.
     * The result is an array containing the output from each cell in the layer.
     * This implements parallel processing of inputs across multiple neurons.
     * 
     * @param array<float> $values Input values to process (must be non-empty)
     * 
     * @return array<float> Array of outputs from all cells in the layer
     * 
     * @throws ValidationException When input array is empty
     * @throws LayerException When any cell calculation fails
     * 
     * @example
     * ```php
     * $layer = new Layer("layer1", [$cell1, $cell2, $cell3]);
     * $input = [1.0, 0.5, -0.2];
     * $outputs = $layer->Calculate($input); // Returns [output1, output2, output3]
     * ```
     */
    public function Calculate(array $values): array {
        if (empty($values)) {
            throw ValidationException::invalidValue('input_values', 'empty array', 'non-empty array');
        }
        
        $result = [];
        foreach ($this->cells as $index => $cell) {
            try {
                $result[] = $cell->Calculate($values);
            } catch (\Exception $e) {
                throw LayerException::operationError(
                    'calculate',
                    "Failed to calculate cell {$index} in layer '{$this->name}': " . $e->getMessage()
                );
            }
        }
        return $result;
    }

    /**
     * Factory method to create a layer with uniform cells.
     * 
     * Creates a new layer with the specified number of cells, each initialized
     * with random weights and default activation functions. All cells will have
     * the same number of weights equal to the layer size. Useful for creating
     * standard layer architectures quickly.
     * 
     * @param int $size Number of cells to create in the layer (must be positive)
     * 
     * @return Layer New layer with uniformly initialized cells
     * 
     * @throws LayerException When size is not positive
     * 
     * @example
     * ```php
     * $layer = Layer::Create(5); // Creates layer with 5 cells, each having 5 weights
     * ```
     */
    public static function Create(int $size): Layer {
        // Validate size
        if ($size <= 0) {
            throw LayerException::invalidSize($size, 'layer size must be positive');
        }
        
        $name = \mc\uuid();
        $cells = [];
        for( $i = 0; $i < $size; $i++ ) {
            $cells[] = new Cell(\mc\uuid(), \mc\array_create($size));
        }
        return new Layer($name, $cells);
    }

    /**
     * Prints layer information to console for debugging purposes.
     * 
     * Outputs the layer's name and information about all its cells
     * in a human-readable format. Useful for debugging and inspection
     * of neural network state during development.
     * 
     * @return void
     * 
     * @example
     * ```php
     * $layer->Print();
     * // Output: Layer: hidden_layer_1
     * //         cell1 (ReLU): 0.5 -0.3 0.8
     * //         cell2 (ReLU): 0.1 0.7 -0.2
     * ```
     */
    public function Print(): void {
        echo "Layer: {$this->name}" . PHP_EOL;
        foreach ($this->cells as $cell) {
            $cell->Print();
        }
    }

    /**
     * Creates a Layer instance from JSON data (deserialization).
     * 
     * Reconstructs a layer from its JSON representation, restoring the name
     * and all cells within the layer. Each cell is also reconstructed from
     * its JSON data including weights and activation functions.
     * 
     * @param string $json JSON string containing layer data (must be non-empty)
     * 
     * @return Layer Reconstructed layer instance
     * 
     * @throws SerializationException When JSON is invalid or cell data is malformed
     * @throws ValidationException When loaded data is invalid
     * 
     * @example
     * ```php
     * $json = '{"name":"layer1","cells":[{"name":"cell1","dna":[0.5,-0.3]}]}';
     * $layer = Layer::Load($json);
     * ```
     */
    public static function Load(string $json): Layer {
        if (empty(trim($json))) {
            throw SerializationException::invalidFormat('JSON string', 'empty string');
        }
        
        $data = json_decode($json, true);
        
        if ($data === null) {
            throw SerializationException::jsonError('decode', json_last_error());
        }
        
        if (!isset($data['name']) || !isset($data['cells'])) {
            throw SerializationException::invalidFormat(
                'Layer JSON with name and cells fields',
                'JSON missing required fields'
            );
        }
        
        $name = $data["name"];
        $cells = [];
        
        foreach ($data["cells"] as $index => $cellData) {
            if (!isset($cellData['name']) || !isset($cellData['dna'])) {
                throw SerializationException::invalidFormat(
                    'Cell JSON with name and dna fields',
                    "Cell at index {$index} missing required fields"
                );
            }
            
            try {
                $cells[] = new Cell($cellData["name"], $cellData["dna"]);
            } catch (\Exception $e) {
                throw SerializationException::corruptedData(
                    'cell data',
                    "Failed to create cell at index {$index}: " . $e->getMessage()
                );
            }
        }
        
        return new Layer($name, $cells);
    }

    /**
     * Serializes the layer to JSON format for persistence or transmission.
     * 
     * Converts the layer's state (name and all cells) into a JSON string.
     * Each cell is serialized with its name and weights. This enables
     * saving trained layers and restoring them later.
     * 
     * @return string JSON representation of the layer
     * 
     * @throws SerializationException When JSON encoding fails
     * 
     * @example
     * ```php
     * $layer = new Layer("layer1", [$cell1, $cell2]);
     * $json = $layer->Json();
     * // Returns: {"name":"layer1","cells":[{cell1_data},{cell2_data}]}
     * ```
     */
    public function Json(): string {
        $data = [
            "name" => $this->name,
            "cells" => []
        ];
        
        foreach ($this->cells as $cell) {
            $data["cells"][] = [
                "name"=> $cell->GetName(),
                "dna" => $cell->GetDna()
            ];
        }
        
        $json = json_encode($data);
        
        if ($json === false) {
            throw SerializationException::jsonError('encode', json_last_error());
        }
        
        return $json;
    }
}