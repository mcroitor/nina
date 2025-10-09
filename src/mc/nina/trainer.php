<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\TrainingException;
use mc\nina\exception\ComputationException;
use function mc\manhattan_distance;

/**
 * Neural network trainer for genetic algorithm-based learning
 * 
 * Creates a population of networks based on a base network and evolutionarily
 * improves their performance based on training data. Uses Manhattan distance
 * to evaluate prediction quality.
 * 
 * @package mc\nina
 * @author Your Name
 * @version 1.0
 * 
 * @example
 * ```php
 * $network = new Network([new Layer([new Cell("input", [0.5, 0.3])])]);
 * $trainer = new Trainer($network, 10);
 * $trainer->Train("hello", "world");
 * $bestNetwork = $trainer->getBestNetwork();
 * ```
 */
class Trainer
{
    private array $networks = [];
    private Network $best_network;

    /**
     * Creates a new trainer with a population of networks based on the source network
     * 
     * Initializes the trainer with the specified network as the best and creates
     * a population of mutated copies of this network for evolutionary learning.
     * 
     * @param Network $network Base neural network for training
     * @param int $pull Population size (1-1000)
     * 
     * @throws ValidationException If the network is not a Network instance
     * @throws TrainingException If the population size is incorrect
     * 
     * @example
     * ```php
     * $network = new Network([new Layer([new Cell("test", [0.5])])]);
     * $trainer = new Trainer($network, 50); // Creates population of 50 networks
     * ```
     */
    public function __construct(Network $network, int $pull)
    {
        // Validate network
        if (!($network instanceof Network)) {
            throw ValidationException::invalidValue(
                'network',
                $network,
                'Network instance',
                "got " . gettype($network)
            );
        }
        
        // Validate pull size
        if ($pull <= 0) {
            throw TrainingException::invalidParameters('pull', $pull, 'must be positive');
        }
        
        if ($pull > 1000) {
            throw TrainingException::invalidParameters('pull', $pull, 'too large (max 1000)');
        }
        
        $this->best_network = $network;
        for ($i = 0; $i < $pull; ++$i) {
            $this->networks[] = $network->Born(5);
        }
    }

    /**
     * Returns all networks in the population
     * 
     * @return array<Network> Array of all networks in the population
     * 
     * @example
     * ```php
     * $trainer = new Trainer($network, 10);
     * $networks = $trainer->getNetworks(); // Returns array of 10 networks
     * ```
     */
    public function getNetworks(): array
    {
        return $this->networks;
    }

    /**
     * Returns a network by index from the population
     * 
     * @param int $index Network index in the population (from 0 to number of networks - 1)
     * 
     * @return Network Network at the specified index
     * 
     * @throws ValidationException If the index is outside the valid range
     * 
     * @example
     * ```php
     * $trainer = new Trainer($network, 10);
     * $firstNetwork = $trainer->getNetworkByIndex(0);
     * $lastNetwork = $trainer->getNetworkByIndex(9);
     * ```
     */
    public function getNetworkByIndex(int $index): Network
    {
        if ($index < 0 || $index >= count($this->networks)) {
            throw ValidationException::outOfRange(
                'network_index',
                $index,
                0,
                count($this->networks) - 1
            );
        }
        return $this->networks[$index];
    }

    /**
     * Tests network performance on input/expected data pair
     * 
     * Uses Manhattan distance to measure the difference between
     * predicted and expected results. Lower value means better performance.
     * 
     * @param Network $network Network to test
     * @param string $input Input data for prediction
     * @param string $expected Expected result
     * 
     * @return float Manhattan distance between predicted and expected result
     * 
     * @throws ComputationException If computation result is invalid
     * @throws TrainingException If testing failed
     */
    private static function Test(Network $network, string $input, string $expected): float
    {
        try {
            $predictor = new Predictor($network);
            $calculated = $predictor->Predict($input);
            $result = manhattan_distance(
                $calculated,
                $expected
            );
            
            if (!is_finite($result)) {
                throw ComputationException::invalidNumericalValue(
                    $result,
                    'distance calculation result'
                );
            }
            
            return $result;
        } catch (\Exception $e) {
            throw TrainingException::invalidTrainingData(
                "Failed to test network: " . $e->getMessage(),
                ['input' => $input, 'expected' => $expected]
            );
        }
    }

    /**
     * Trains the population of networks on input/expected data pair
     * 
     * Tests all networks in the population and selects the best one based on results.
     * Then creates a new population based on the best network with mutations.
     * The process repeats for evolutionary performance improvement.
     * 
     * @param string $input Input data for training
     * @param string $expected Expected result
     * 
     * @throws TrainingException If input data is invalid or training failed
     * 
     * @example
     * ```php
     * $trainer = new Trainer($network, 20);
     * $trainer->Train("hello", "world");
     * $trainer->Train("test", "data");
     * $bestNetwork = $trainer->getBestNetwork();
     * ```
     */
    public function Train(string $input, string $expected): void
    {
        // Validate inputs
        if (empty($input)) {
            throw TrainingException::invalidTrainingData('input string cannot be empty');
        }
        
        if (empty($expected)) {
            throw TrainingException::invalidTrainingData('expected string cannot be empty');
        }
        
        try {
            $best = $this->best_network;
            $bestScore = self::Test($best, $input, $expected);
            
            foreach ($this->networks as $index => $network) {
                try {
                    $score = self::Test($network, $input, $expected);
                    if ($score < $bestScore) {
                        $best = $network;
                        $bestScore = $score;
                    }
                } catch (\Exception $e) {
                    throw TrainingException::convergenceError(
                        $index,
                        $bestScore,
                        "Failed to test network {$index}: " . $e->getMessage()
                    );
                }
            }
            
            $this->best_network = $best;
            $this->networks = [];
            
            for ($i = 0; $i < count($this->networks); ++$i) {
                $this->networks[] = $this->best_network->Born(10);
            }
        } catch (TrainingException $e) {
            throw $e;
        } catch (\Exception $e) {
            throw TrainingException::invalidTrainingData(
                "Training failed: " . $e->getMessage(),
                ['input' => $input, 'expected' => $expected]
            );
        }
    }

    /**
     * Returns the best network from the population
     * 
     * After each training cycle, the best network is updated based on
     * testing results of the entire population.
     * 
     * @return Network Current best neural network
     * 
     * @example
     * ```php
     * $trainer = new Trainer($network, 15);
     * $trainer->Train("input", "output");
     * $bestNetwork = $trainer->getBestNetwork();
     * $predictor = new Predictor($bestNetwork);
     * ```
     */
    public function getBestNetwork(): Network
    {
        return $this->best_network;
    }
}
