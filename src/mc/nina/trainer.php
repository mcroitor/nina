<?php

namespace mc\nina;

use mc\nina\exception\ValidationException;
use mc\nina\exception\TrainingException;
use mc\nina\exception\ComputationException;

use function mc\euclidean_distance;

/**
 * Neural network trainer for genetic algorithm-based learning
 * 
 * Creates a population of networks based on a base network and evolutionarily
 * improves their performance based on training data. Uses Manhattan distance
 * to evaluate prediction quality.
 * 
 * @package mc\nina
 * @author Mihail Croitor
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
    private int $mutationRate;

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
    public function __construct(Network $network, int $pull, int|float $mutationRate = 10)
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
        
        if (is_float($mutationRate) && $mutationRate > 0.0 && $mutationRate < 1.0) {
            $normalizedRate = (int)round($mutationRate * 100);
        } else {
            $normalizedRate = (int)round($mutationRate);
        }
        $this->mutationRate = max(0, min(100, $normalizedRate));
        $this->best_network = $network;
        for ($i = 0; $i < $pull; ++$i) {
            $this->networks[] = $network->Born($this->mutationRate);
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
    private static function Test(Network $network, string|array $input, string|array $expected): float
    {
        try {
            if (is_array($input) && is_array($expected)) {
                $calculated = $network->Calculate($input);
                $result = euclidean_distance($calculated, $expected);
            } else {
                $inputStr = is_array($input) ? \mc\array2string($input) : $input;
                $expectedStr = is_array($expected) ? \mc\array2string($expected) : $expected;
                $predictor = new Predictor($network);
                $calculated = $predictor->Predict($inputStr);
                $result = euclidean_distance(
                    $calculated,
                    $expectedStr
                );
            }
            
            if (!is_finite($result)) {
                throw ComputationException::invalidNumericalValue(
                    $result,
                    'distance calculation result'
                );
            }

            if (defined('__DEBUG__') && __DEBUG__) {
                echo "[DEBUG] Distance: {$result}\n";
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
     * @param string|array $input Input data for training
     * @param string|array $expected Expected result
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
    public function Train(string|array $input, string|array $expected): void
    {
        // Validate inputs
        if (empty($input)) {
            throw TrainingException::invalidTrainingData('input cannot be empty');
        }
        
        if (empty($expected)) {
            throw TrainingException::invalidTrainingData('expected cannot be empty');
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
            $populationSize = count($this->networks);
            $this->networks = [];
            
            for ($i = 0; $i < $populationSize; ++$i) {
                $this->networks[] = $this->best_network->Born($this->mutationRate);
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
     * Trains the population of networks on a batch of samples in a single generation.
     *
     * Evaluates each network across all samples in the batch, selects the network
     * with the lowest total distance, and spawns the next generation once.
     *
     * @param array $samples Array of samples (either ['input' => ..., 'target' => ...] or [$input, $expected])
     * @throws TrainingException If batch data is empty or invalid
     */
    public function TrainBatch(array $samples): void
    {
        if (empty($samples)) {
            throw TrainingException::invalidTrainingData('training batch cannot be empty');
        }

        $pairs = [];
        foreach ($samples as $index => $sample) {
            if (isset($sample['input']) && isset($sample['target'])) {
                $pairs[] = [$sample['input'], $sample['target']];
            } elseif (isset($sample[0]) && isset($sample[1])) {
                $pairs[] = [$sample[0], $sample[1]];
            } else {
                throw TrainingException::invalidTrainingData("Invalid sample format at index {$index}");
            }
        }

        // Select a representative evaluation subset when the batch is large for fast screening
        $screeningPairs = $pairs;
        if (count($pairs) > 8) {
            $screeningPairs = array_slice($pairs, 0, 8);
        }

        $evaluateSubset = function (Network $net, array $evalPairs): float {
            $totalScore = 0.0;
            foreach ($evalPairs as [$in, $exp]) {
                $totalScore += self::Test($net, $in, $exp);
            }
            return $totalScore / count($evalPairs);
        };

        // Include a memetic (guided variation) candidate in the population if inputs are numeric arrays
        if (!empty($this->networks) && is_array($pairs[0][0]) && is_array($pairs[0][1])) {
            $guidedCandidate = $this->best_network->Born(max(1, (int)($this->mutationRate / 2)));
            $guidedTrainer = new BackpropagationTrainer(
                $guidedCandidate,
                new \mc\nina\loss\MeanSquaredError(),
                max(0.001, min(0.05, $this->mutationRate / 1000.0)),
                new \mc\nina\optimizer\Adam()
            );
            $guidedTrainer->trainBatch($screeningPairs, 1);
            $this->networks[0] = $guidedTrainer->getTrainedNetwork();
        }

        $best = $this->best_network;
        $bestScore = $evaluateSubset($best, $screeningPairs);

        foreach ($this->networks as $network) {
            $score = $evaluateSubset($network, $screeningPairs);
            if ($score < $bestScore) {
                $best = $network;
                $bestScore = $score;
            }
        }

        $this->best_network = $best;
        $populationSize = count($this->networks);
        $this->networks = [];

        for ($i = 0; $i < $populationSize; ++$i) {
            $this->networks[] = $this->best_network->Born($this->mutationRate);
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

    public function getMutationRate(): int
    {
        return $this->mutationRate;
    }
}
