<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\NetworkException;
use mc\nina\exception\SerializationException;

/**
 * Complete neural network composed of sequential layers.
 * 
 * The Network class represents a full neural network with multiple layers
 * connected sequentially. Data flows from input through hidden layers to output.
 * Supports forward propagation, genetic algorithm training, and serialization.
 * 
 * @package mc\nina
 * @author NiNa Library
 * @version 0.2.0
 * @since 0.1.0
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
     * @param string $name Unique identifier for this network (must be non-empty)
     * @param array<Layer> $layers Array of Layer objects (must be non-empty)
     * 
     * @throws ValidationException When name is empty or layers contain invalid types
     * @throws NetworkException When layers array is empty
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

    public static function Create(string $name, int $nrLayers, int $layerSize): Network {
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
            $layers[] = Layer::Create($layerSize);
        }
        return new Network($name, $layers);
    }

    public function GetName(): string {
        return $this->name;
    }

    public function GetNrLayers(): int {
        return count($this->layers);
    }

    public function GetLayerSize(): int {
        return $this->layers[0]->GetSize();
    }

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
     * @param array<float> $values Input values to process
     * @return array<float> Final network outputs
     * @throws ValidationException When input is empty
     * @throws NetworkException When layer calculation fails
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

    public function Print(): void {
        echo "Network: {$this->name}" . PHP_EOL;
        foreach ($this->layers as $layer) {
            $layer->Print();
        }
    }

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