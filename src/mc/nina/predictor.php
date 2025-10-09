<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;
use function mc\string2array;
use function mc\array2string;

/**
 * Predictor based on a trained neural network
 * 
 * Wrapper around a neural network that converts string input data
 * to numeric arrays, performs prediction and converts the result back to string.
 * Provides convenient interface for using trained neural networks.
 * 
 * @package mc\nina
 * @author Your Name
 * @version 1.0
 * 
 * @example
 * ```php
 * $network = new Network([new Layer([new Cell("output", [0.5, 0.3])])]);
 * $predictor = new Predictor($network);
 * $result = $predictor->Predict("hello world");
 * echo $result; // Outputs predicted string
 * ```
 */
class Predictor{
    private Network $network;

    /**
     * Creates a new predictor with the specified neural network
     * 
     * Initializes the predictor with a trained neural network
     * that will be used to perform predictions.
     * 
     * @param Network $network Trained neural network for predictions
     * 
     * @throws ValidationException If the passed network is not a Network instance
     * 
     * @example
     * ```php
     * $trainedNetwork = $trainer->getBestNetwork();
     * $predictor = new Predictor($trainedNetwork);
     * ```
     */
    public function __construct(Network $network){
        // Validate network
        if (!($network instanceof Network)) {
            throw ValidationException::invalidValue(
                'network',
                $network,
                'Network instance',
                "got " . gettype($network)
            );
        }
        
        $this->network = $network;
    }

    /**
     * Performs prediction based on input string
     * 
     * Converts input string to numeric array, passes it to the neural network
     * for computation, gets the result and converts it back to string.
     * Validates results at each stage.
     * 
     * @param string $input Input string for prediction
     * 
     * @return string Predicted string based on neural network computations
     * 
     * @throws ValidationException If input string is empty
     * @throws ComputationException If prediction failed or result is invalid
     * 
     * @example
     * ```php
     * $predictor = new Predictor($trainedNetwork);
     * $result = $predictor->Predict("Hello, world!");
     * echo "Prediction: " . $result;
     * ```
     */
    public function Predict(string $input): string{
        // Validate input
        if (empty($input)) {
            throw ValidationException::invalidValue('input', 'empty string', 'non-empty string');
        }
        
        try {
            $values = string2array($input);
            $result = $this->network->Calculate($values);
            
            // Validate result array before conversion
            foreach ($result as $index => $value) {
                if (!is_finite($value)) {
                    throw ComputationException::invalidNumericalValue(
                        $value,
                        "prediction result at index {$index}"
                    );
                }
            }
            
            return array2string($result);
        } catch (ComputationException $e) {
            throw $e;
        } catch (\Exception $e) {
            throw ComputationException::domainError(
                'prediction',
                $input,
                $e->getMessage()
            );
        }
    }
}