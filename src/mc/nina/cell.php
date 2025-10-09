<?php

namespace mc\nina;

use mc\nina\activation\ActivationFunction;
use mc\nina\activation\ReLU;
use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;
use mc\nina\exception\SerializationException;

/**
 * Neural cell (neuron) implementation with weights, activation function and forward propagation.
 * 
 * The Cell class represents a single neuron in a neural network with configurable weights (DNA)
 * and an activation function. It performs forward propagation calculations and supports
 * genetic algorithm operations like mutation.
 * 
 * @package mc\nina
 * @author NiNa Library
 * @version 0.2.0
 * @since 0.1.0
 */
class Cell
{
    /**
     * Unique name identifier for this cell
     * @var string
     */
    private string $name;
    
    /**
     * Neural weights (DNA) - values between 0 and 1 representing synaptic strengths
     * @var array<float>
     */
    private array $dna;
    
    /**
     * Activation function for this cell
     * @var ActivationFunction
     */
    private ActivationFunction $activationFunction;

    /**
     * Constructs a new neural cell with specified weights and activation function.
     * 
     * Creates a neural cell with the given name, weight array (DNA), and activation function.
     * The weights determine the strength of connections from input signals to this neuron.
     * 
     * @param string $name Unique identifier for this cell (must be non-empty)
     * @param array<float> $dna Array of weights/DNA values (must be non-empty, finite numbers)
     * @param ActivationFunction|null $activationFunction Activation function (defaults to ReLU if null)
     * 
     * @throws ValidationException When name is empty or DNA array is invalid
     * @throws ComputationException When DNA contains non-finite values (NaN, Infinity)
     * 
     * @example
     * ```php
     * use mc\nina\Cell;
     * use mc\nina\activation\Sigmoid;
     * 
     * $cell = new Cell("neuron1", [0.5, -0.3, 0.8], new Sigmoid());
     * ```
     */
    public function __construct(string $name, array $dna, ?ActivationFunction $activationFunction = null)
    {
        // Validate name
        if (empty(trim($name))) {
            throw ValidationException::invalidValue('name', $name, 'non-empty string');
        }
        
        // Validate DNA array
        if (empty($dna)) {
            throw ValidationException::invalidValue('dna', 'empty array', 'non-empty array');
        }
        
        // Validate DNA values
        foreach ($dna as $index => $value) {
            if (!is_numeric($value)) {
                throw ValidationException::invalidValue(
                    "dna[{$index}]", 
                    $value, 
                    'numeric',
                    "got " . gettype($value)
                );
            }
            
            // Check for invalid numerical values
            if (!is_finite((float)$value)) {
                throw ComputationException::invalidNumericalValue(
                    $value, 
                    "DNA value at index {$index}"
                );
            }
            
            // Convert to float for consistency
            $dna[$index] = (float)$value;
        }
        
        $this->name = trim($name);
        $this->dna = $dna;
        $this->activationFunction = $activationFunction ?? new ReLU(); // Default to ReLU
    }

    /**
     * Factory method to create a cell with uniform weight initialization.
     * 
     * Creates a new neural cell with all weights initialized to the same value.
     * This is useful for creating cells with predictable initial states or
     * for testing purposes.
     * 
     * @param string $name Unique identifier for the cell (must be non-empty)
     * @param int $size Number of weights/inputs for this cell (must be positive)
     * @param float $value Initial value for all weights (must be finite, defaults to 0.0)
     * @param ActivationFunction|null $activationFunction Activation function (defaults to ReLU if null)
     * 
     * @return Cell New cell instance with uniform weights
     * 
     * @throws ValidationException When name is empty or size is not positive
     * @throws ComputationException When value is not finite (NaN, Infinity)
     * 
     * @example
     * ```php
     * // Create cell with 5 weights all set to 0.1
     * $cell = Cell::Create("input_neuron", 5, 0.1);
     * 
     * // Create cell with ReLU activation
     * $cell = Cell::Create("hidden_neuron", 3, 0.5, new ReLU());
     * ```
     */
    public static function Create(string $name, int $size, float $value = 0., ?ActivationFunction $activationFunction = null): Cell
    {
        // Validate parameters
        if (empty(trim($name))) {
            throw ValidationException::invalidValue('name', $name, 'non-empty string');
        }
        
        if ($size <= 0) {
            throw ValidationException::outOfRange('size', $size, 1);
        }
        
        if (!is_finite($value)) {
            throw ComputationException::invalidNumericalValue($value, 'default value');
        }
        
        $dna = \mc\array_create($size, $value);
        return new Cell($name, $dna, $activationFunction);
    }

    /**
     * Gets the unique name identifier of this cell.
     * 
     * @return string The cell's name/identifier
     */
    public function GetName(): string
    {
        return $this->name;
    }

    /**
     * Gets the complete DNA (weights) array of this cell.
     * 
     * Returns a copy of the internal weights array representing the synaptic
     * strengths of this neuron. These weights are used in forward propagation
     * calculations.
     * 
     * @return array<float> Array of weights/DNA values
     */
    public function GetDna(): array
    {
        return $this->dna;
    }

    /**
     * Gets the number of weights (input connections) in this cell.
     * 
     * Returns the size of the DNA array, which represents how many
     * input connections this neuron can accept.
     * 
     * @return int Number of weights/inputs
     */
    public function GetSize(): int
    {
        return count($this->dna);
    }

    /**
     * Gets a specific weight value by index.
     * 
     * Retrieves the weight at the specified index from the DNA array.
     * Used for accessing individual synaptic strengths.
     * 
     * @param int $index Index of the weight to retrieve (0-based)
     * 
     * @return float The weight value at the specified index
     * 
     * @throws ValidationException When index is out of bounds
     */
    public function GetValue(int $index): float
    {
        if ($index < 0 || $index >= count($this->dna)) {
            throw ValidationException::outOfRange('index', $index, 0, count($this->dna) - 1);
        }
        return $this->dna[$index];
    }

    /**
     * Gets the current activation function of this cell.
     * 
     * Returns the activation function instance currently used by this cell
     * for output computation. This can be any implementation of ActivationFunction
     * interface (ReLU, Sigmoid, Tanh, Softmax, etc.).
     * 
     * @return ActivationFunction The current activation function instance
     */
    public function GetActivationFunction(): ActivationFunction
    {
        return $this->activationFunction;
    }

    /**
     * Sets a new activation function for this cell.
     * 
     * Changes the activation function used by this cell. This affects
     * all future calculations performed by the Calculate() method.
     * Allows dynamic switching between different activation functions.
     * 
     * @param ActivationFunction $activationFunction The new activation function to use
     * 
     * @return void
     * 
     * @example
     * ```php
     * $cell = new Cell("neuron1", [0.5, -0.3]);
     * $cell->SetActivationFunction(new Sigmoid()); // Switch to Sigmoid
     * ```
     */
    public function SetActivationFunction(ActivationFunction $activationFunction): void
    {
        $this->activationFunction = $activationFunction;
    }

    /**
     * Creates a mutated offspring cell based on current cell (genetic algorithm operation).
     * 
     * Generates a new cell with randomly mutated weights while preserving the activation function.
     * Each weight is modified by a random amount based on the mutation percentage.
     * The new cell gets a unique UUID name to distinguish it from the parent.
     * 
     * This method is essential for genetic algorithm training where the population
     * evolves through mutation and selection.
     * 
     * @param int $percent Mutation percentage (0-100) determining how much weights can change
     * 
     * @return Cell New cell with mutated weights and same activation function
     * 
     * @throws ValidationException When percent is negative or exceeds reasonable bounds
     * 
     * @example
     * ```php
     * $parent = new Cell("parent", [0.5, -0.3, 0.8]);
     * $child = $parent->Born(10); // 10% mutation rate
     * // Child will have similar but slightly different weights
     * ```
     */
    public function Born(int $percent): Cell
    {
        $name = \mc\uuid();
        $dna = [];
        foreach ($this->dna as $value) {
            $dna[] = \mc\mutate($value, $percent);
        }
        return new Cell($name, $dna, $this->activationFunction);
    }

    /**
     * Performs forward propagation through this neural cell.
     * 
     * Calculates the weighted sum of inputs and applies the activation function.
     * This is the core computation of a neural cell: sum(input[i] * weight[i]) -> activation_function.
     * Input array size must match the number of weights (DNA size).
     * 
     * @param array<float> $values Input values to process (must be non-empty)
     * 
     * @return float Activated output value after applying activation function
     * 
     * @throws ValidationException When input array is empty or size doesn't match DNA size
     * @throws ComputationException When input contains non-finite values or computation fails
     * 
     * @example
     * ```php
     * $cell = new Cell("neuron1", [0.5, -0.3, 0.8]);
     * $input = [1.0, 0.5, -0.2];
     * $output = $cell->Calculate($input); // Returns activated result
     * ```
     */
    public function Calculate(array $values): float
    {
        // Validate input size
        $inputSize = count($values);
        $dnaSize = count($this->dna);
        
        if ($inputSize === 0) {
            throw ValidationException::invalidValue('input_values', 'empty array', 'non-empty array');
        }
        
        if ($inputSize !== $dnaSize) {
            throw ValidationException::invalidArraySize(
                'input_values',
                $inputSize,
                $dnaSize
            );
        }
        
        // Calculate weighted sum (dot product)
        $weightedSum = 0.0;
        for ($i = 0; $i < $inputSize; $i++) {
            // Validate that input values are numeric
            if (!is_numeric($values[$i])) {
                throw ValidationException::invalidValue(
                    "input_values[{$i}]",
                    $values[$i],
                    'numeric',
                    "got " . gettype($values[$i])
                );
            }
            
            $value = (float)$values[$i];
            
            // Check for invalid numerical values
            if (!is_finite($value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "input value at index {$i}"
                );
            }
            
            $weightedSum += $value * $this->dna[$i];
        }
        
        // Check for numerical overflow in weighted sum
        if (!is_finite($weightedSum)) {
            throw ComputationException::invalidNumericalValue(
                $weightedSum,
                "weighted sum calculation in cell '{$this->name}'"
            );
        }
        
        // Apply activation function
        try {
            return $this->activationFunction->activate($weightedSum);
        } catch (\Exception $e) {
            throw ComputationException::domainError(
                'activation function',
                $weightedSum,
                $e->getMessage()
            );
        }
    }

    /**
     * Calculates raw weighted sum without applying activation function.
     * 
     * Computes the dot product of input values and weights (DNA) without activation.
     * This is useful for debugging, analysis, or when you need the pre-activation value.
     * Formula: sum(input[i] * weight[i]) for i in 0..length-1
     * 
     * @param array<float> $values Input values to multiply with weights (must match DNA size)
     * 
     * @return float Raw weighted sum before activation function
     * 
     * @throws ValidationException When input array is empty or size doesn't match DNA size
     * @throws ComputationException When input contains non-finite values
     * 
     * @example
     * ```php
     * $cell = new Cell("neuron1", [0.5, -0.3, 0.8]);
     * $input = [1.0, 0.5, -0.2];
     * $raw = $cell->CalculateRaw($input); // Returns: 1.0*0.5 + 0.5*(-0.3) + (-0.2)*0.8 = 0.19
     * ```
     */
    public function CalculateRaw(array $values): float
    {
        // Validate input size
        $inputSize = count($values);
        $dnaSize = count($this->dna);
        
        if ($inputSize === 0) {
            throw ValidationException::invalidValue('input_values', 'empty array', 'non-empty array');
        }
        
        if ($inputSize !== $dnaSize) {
            throw ValidationException::invalidArraySize(
                'input_values',
                $inputSize,
                $dnaSize
            );
        }
        
        // Calculate weighted sum (dot product)
        $result = 0.0;
        for ($i = 0; $i < $inputSize; $i++) {
            // Validate that input values are numeric
            if (!is_numeric($values[$i])) {
                throw ValidationException::invalidValue(
                    "input_values[{$i}]",
                    $values[$i],
                    'numeric',
                    "got " . gettype($values[$i])
                );
            }
            
            $value = (float)$values[$i];
            
            // Check for invalid numerical values
            if (!is_finite($value)) {
                throw ComputationException::invalidNumericalValue(
                    $value,
                    "input value at index {$i}"
                );
            }
            
            $result += $value * $this->dna[$i];
        }
        
        // Check for numerical overflow in result
        if (!is_finite($result)) {
            throw ComputationException::invalidNumericalValue(
                $result,
                "raw calculation result in cell '{$this->name}'"
            );
        }
        
        return $result;
    }

    /**
     * Prints cell information to console for debugging purposes.
     * 
     * Outputs the cell's name, activation function, and all weight values
     * in a human-readable format. Useful for debugging and inspection
     * of neural network state during development.
     * 
     * @return void
     * 
     * @example
     * ```php
     * $cell = new Cell("neuron1", [0.5, -0.3, 0.8]);
     * $cell->Print();
     * // Output: neuron1 (ReLU): 0.5 -0.3 0.8
     * ```
     */
    public function Print(): void
    {
        echo "{$this->name} ({$this->activationFunction->getName()}): ";
        foreach ($this->dna as $value) {
            echo "{$value} ";
        }
        echo PHP_EOL;
    }

    /**
     * Creates a Cell instance from JSON data (deserialization).
     * 
     * Reconstructs a cell from its JSON representation, restoring the name,
     * weights, and activation function. The activation function is determined
     * by the stored name and mapped to the appropriate class. If no activation
     * function is specified, defaults to ReLU.
     * 
     * @param string $json JSON string containing cell data (must be non-empty)
     * 
     * @return Cell Reconstructed cell instance
     * 
     * @throws SerializationException When JSON is invalid or activation function is unknown
     * @throws ValidationException When loaded data is invalid
     * 
     * @example
     * ```php
     * $json = '{"name":"neuron1","dna":[0.5,-0.3,0.8],"activation":"ReLU"}';
     * $cell = Cell::Load($json);
     * ```
     */
    public static function Load(string $json): Cell
    {
        if (empty(trim($json))) {
            throw SerializationException::invalidFormat('JSON string', 'empty string');
        }
        
        $data = json_decode($json, true);
        
        if ($data === null) {
            throw SerializationException::jsonError('decode', json_last_error());
        }
        
        if (!isset($data['name']) || !isset($data['dna'])) {
            throw SerializationException::invalidFormat(
                'Cell JSON with name and dna fields',
                'JSON missing required fields'
            );
        }
        
        // Load activation function if specified, otherwise use default (ReLU)
        $activationFunction = null;
        if (isset($data["activation"])) {
            try {
                $activationFunction = self::CreateActivationFromName($data["activation"]);
            } catch (\Exception $e) {
                throw SerializationException::invalidFormat(
                    'valid activation function name',
                    $data["activation"],
                    $e->getMessage()
                );
            }
        }
        
        return new Cell($data["name"], $data["dna"], $activationFunction);
    }

    /**
     * Create activation function instance from name
     * 
     * @param string $name Name of the activation function
     * @return ActivationFunction The activation function instance
     */
    private static function CreateActivationFromName(string $name): ActivationFunction
    {
        switch (strtolower($name)) {
            case 'sigmoid':
                return new \mc\nina\activation\Sigmoid();
            case 'tanh':
                return new \mc\nina\activation\Tanh();
            case 'relu':
                return new \mc\nina\activation\ReLU();
            case 'softmax':
                return new \mc\nina\activation\Softmax();
            default:
                return new \mc\nina\activation\ReLU(); // Default fallback
        }
    }

    /**
     * Serializes the cell to JSON format for persistence or transmission.
     * 
     * Converts the cell's state (name, weights, activation function) into a JSON string.
     * The activation function is stored by name and can be reconstructed during loading.
     * This enables saving trained models and restoring them later.
     * 
     * @return string JSON representation of the cell
     * 
     * @throws SerializationException When JSON encoding fails
     * 
     * @example
     * ```php
     * $cell = new Cell("neuron1", [0.5, -0.3, 0.8]);
     * $json = $cell->Json();
     * // Returns: {"name":"neuron1","dna":[0.5,-0.3,0.8],"activation":"ReLU"}
     * ```
     */
    public function Json(): string
    {
        $data = [
            "name" => $this->name,
            "dna" => $this->dna,
            "activation" => $this->activationFunction->getName()
        ];
        
        $json = json_encode($data);
        
        if ($json === false) {
            throw SerializationException::jsonError('encode', json_last_error());
        }
        
        return $json;
    }
}
