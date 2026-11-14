<?php

namespace mc\nina;

include_once __DIR__ . '/optimizer/Optimizer.php';
include_once __DIR__ . '/optimizer/AbstractOptimizer.php';
include_once __DIR__ . '/optimizer/SGD.php';
include_once __DIR__ . '/optimizer/Adam.php';
include_once __DIR__ . '/optimizer/RMSprop.php';
include_once __DIR__ . '/schedule/LearningRateSchedule.php';
include_once __DIR__ . '/schedule/ConstantSchedule.php';
include_once __DIR__ . '/schedule/StepDecaySchedule.php';
include_once __DIR__ . '/schedule/ExponentialSchedule.php';
include_once __DIR__ . '/schedule/PlateauSchedule.php';

use mc\nina\exception\ValidationException;
use mc\nina\exception\TrainingException;
use mc\nina\exception\ComputationException;
use mc\nina\exception\NetworkException;
use mc\nina\loss\LossFunction;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Optimizer;
use mc\nina\optimizer\SGD;
use mc\nina\schedule\LearningRateSchedule;
use mc\nina\schedule\ConstantSchedule;

/**
 * Neural network trainer using backpropagation algorithm
 * 
 * Implements the backpropagation algorithm for training neural networks through
 * gradient descent. Unlike the genetic algorithm trainer, this trainer uses
 * mathematical gradient computation to optimize network weights.
 * 
 * ## Algorithm Overview
 * 
 * ### Forward Pass
 * 1. Input propagates through network layer by layer
 * 2. Each layer's activations and weighted sums are cached
 * 3. Final output is computed and cached
 * 
 * ### Backward Pass
 * 1. Compute loss between predicted and expected outputs
 * 2. Calculate output layer gradients using loss derivative
 * 3. Propagate gradients backward through hidden layers
 * 4. Update weights using computed gradients and learning rate
 * 
 * ### Training Process
 * - Multiple epochs with forward and backward passes
 * - Learning rate controls step size of weight updates
 * - Loss tracking enables convergence monitoring
 * - Early stopping available based on loss threshold
 * 
 * ## Mathematical Foundation
 * 
 * ### Gradient Computation
 * For layer l with weights W, the gradient is:
 * ∇W[l] = δ[l] × a[l-1]^T
 * 
 * Where δ[l] is the error signal:
 * - Output layer: δ[L] = ∇loss × σ'(z[L])
 * - Hidden layers: δ[l] = (W[l+1]^T × δ[l+1]) ⊙ σ'(z[l])
 * 
 * ### Weight Update Rule
 * W[l] = W[l] - α × ∇W[l]
 * Where α is the learning rate
 * 
 * @package mc\nina
 * @author Mihail Croitor
 * @version 1.0
 * @since 0.3.0
 * 
 * @example Basic training setup:
 * ```php
 * $network = Network::Create("classifier", 3, 10);
 * $lossFunction = new MeanSquaredError();
 * $trainer = new BackpropagationTrainer($network, $lossFunction, 0.01);
 * 
 * // Train on single example
 * $input = [0.1, 0.5, 0.9];
 * $expected = [0.0, 1.0, 0.0];
 * $trainer->trainSingle($input, $expected);
 * 
 * // Get updated network
 * $trainedNetwork = $trainer->getNetwork();
 * ```
 * 
 * @example Batch training with monitoring:
 * ```php
 * $trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.01);
 * 
 * $trainingData = [
 *     [[0.1, 0.2], [0.9]],
 *     [[0.8, 0.3], [0.1]],
 *     [[0.5, 0.7], [0.6]]
 * ];
 * 
 * $losses = $trainer->trainBatch($trainingData, 100); // 100 epochs
 * echo "Final loss: " . end($losses) . "\n";
 * ```
 */
class BackpropagationTrainer
{
    /**
     * Neural network being trained
     * @var Network
     */
    private Network $network;
    
    /**
     * Loss function for computing gradients
     * @var LossFunction
     */
    private LossFunction $lossFunction;
    
    /**
     * Learning rate for gradient descent
     * @var float
     */
    private float $learningRate;

    /**
     * Initial base learning rate
     * @var float
     */
    private float $initialLearningRate;

    /**
     * Weight optimizer (SGD, Adam, RMSprop)
     * @var Optimizer
     */
    private Optimizer $optimizer;

    /**
     * Learning rate decay schedule
     * @var LearningRateSchedule
     */
    private LearningRateSchedule $lrSchedule;
    
    /**
     * Training loss history
     * @var array<float>
     */
    private array $lossHistory = [];
    
    /**
     * Current epoch number
     * @var int
     */
    private int $currentEpoch = 0;

    /**
     * Maximum absolute value for gradient clipping by value
     * @var float|null
     */
    private ?float $clipValue = null;

    /**
     * Maximum Euclidean norm for gradient clipping by norm
     * @var float|null
     */
    private ?float $clipNorm = null;

    /**
     * Creates a new backpropagation trainer
     * 
     * @param Network $network Neural network to train
     * @param LossFunction $lossFunction Loss function for computing gradients
     * @param float $learningRate Learning rate for gradient descent (0.0001-1.0)
     * @param Optimizer|null $optimizer Weight optimizer (defaults to SGD)
     * @param LearningRateSchedule|null $lrSchedule Learning rate schedule (defaults to ConstantSchedule)
     * 
     * @throws ValidationException If parameters are invalid
     * @throws TrainingException If network or loss function setup fails
     * 
     * @example
     * ```php
     * $network = Network::Create("test", 2, 5);
     * $trainer = new BackpropagationTrainer(
     *     $network,
     *     new MeanSquaredError(),
     *     0.001 // Learning rate
     * );
     * ```
     */
    public function __construct(
        Network $network, 
        LossFunction $lossFunction, 
        float $learningRate,
        ?Optimizer $optimizer = null,
        ?LearningRateSchedule $lrSchedule = null
    ) {
        // Validate network
        if (!($network instanceof Network)) {
            throw ValidationException::invalidValue(
                'network',
                $network,
                'Network instance',
                "got " . gettype($network)
            );
        }
        
        // Validate loss function
        if (!($lossFunction instanceof LossFunction)) {
            throw ValidationException::invalidValue(
                'lossFunction',
                $lossFunction,
                'LossFunction instance',
                "got " . gettype($lossFunction)
            );
        }
        
        // Validate learning rate
        if (!is_numeric($learningRate) || !is_finite($learningRate)) {
            throw ValidationException::invalidValue(
                'learningRate',
                $learningRate,
                'finite numeric value',
                "got " . gettype($learningRate)
            );
        }
        
        if ($learningRate <= 0.0) {
            throw ComputationException::invalidNumericalValue(
                $learningRate,
                'learning rate must be positive'
            );
        }
        
        if ($learningRate > 1.0) {
            throw ComputationException::invalidNumericalValue(
                $learningRate,
                'learning rate too large (should be ≤ 1.0)'
            );
        }
        
        $this->network = $network;
        $this->lossFunction = $lossFunction;
        $this->initialLearningRate = $learningRate;
        $this->learningRate = $learningRate;
        $this->optimizer = $optimizer ?? new SGD();
        $this->lrSchedule = $lrSchedule ?? new ConstantSchedule();
        $this->lossHistory = [];
        $this->currentEpoch = 0;
    }

    /**
     * Returns the neural network being trained
     * 
     * @return Network Current network with updated weights
     * 
     * @example
     * ```php
     * $trainer = new BackpropagationTrainer($network, $loss, 0.01);
     * $trainer->trainSingle($input, $target);
     * $updatedNetwork = $trainer->getNetwork();
     * ```
     */
    public function getNetwork(): Network
    {
        return $this->network;
    }

    /**
     * Returns the neural network being trained (alias for getNetwork)
     * 
     * @return Network The neural network
     */
    public function getTrainedNetwork(): Network
    {
        return $this->network;
    }

    /**
     * Returns the loss function used for training
     * 
     * @return LossFunction Current loss function
     * 
     * @example
     * ```php
     * $lossFunction = $trainer->getLossFunction();
     * echo "Using loss: " . $lossFunction->getName() . "\n";
     * ```
     */
    public function getLossFunction(): LossFunction
    {
        return $this->lossFunction;
    }

    /**
     * Returns the current learning rate
     * 
     * @return float Current learning rate value
     * 
     * @example
     * ```php
     * $currentLR = $trainer->getLearningRate();
     * echo "Learning rate: {$currentLR}\n";
     * ```
     */
    public function getLearningRate(): float
    {
        return $this->learningRate;
    }

    /**
     * Sets a new learning rate
     * 
     * @param float $learningRate New learning rate (0.0001-1.0)
     * 
     * @throws ComputationException If learning rate is invalid
     * 
     * @example
     * ```php
     * $trainer->setLearningRate(0.001); // Reduce learning rate
     * ```
     */
    public function setLearningRate(float $learningRate): void
    {
        if (!is_numeric($learningRate) || !is_finite($learningRate)) {
            throw ValidationException::invalidValue(
                'learningRate',
                $learningRate,
                'finite numeric value',
                "got " . gettype($learningRate)
            );
        }
        
        if ($learningRate <= 0.0) {
            throw ComputationException::invalidNumericalValue(
                $learningRate,
                'learning rate must be positive'
            );
        }
        
        if ($learningRate > 1.0) {
            throw ComputationException::invalidNumericalValue(
                $learningRate,
                'learning rate too large (should be ≤ 1.0)'
            );
        }
        
        $this->learningRate = $learningRate;
    }

    /**
     * Returns the optimizer used for weight updates
     * 
     * @return Optimizer
     */
    public function getOptimizer(): Optimizer
    {
        return $this->optimizer;
    }

    /**
     * Sets a new optimizer for weight updates
     * 
     * @param Optimizer $optimizer
     */
    public function setOptimizer(Optimizer $optimizer): void
    {
        $this->optimizer = $optimizer;
    }

    /**
     * Returns the learning rate schedule
     * 
     * @return LearningRateSchedule
     */
    public function getLearningRateSchedule(): LearningRateSchedule
    {
        return $this->lrSchedule;
    }

    /**
     * Sets a learning rate schedule
     * 
     * @param LearningRateSchedule $lrSchedule
     */
    public function setLearningRateSchedule(LearningRateSchedule $lrSchedule): void
    {
        $this->lrSchedule = $lrSchedule;
    }

    /**
     * Sets L2 weight decay regularization factor on current optimizer
     * 
     * @param float $weightDecay
     */
    public function setWeightDecay(float $weightDecay): void
    {
        $this->optimizer->setWeightDecay($weightDecay);
    }

    /**
     * Gets L2 weight decay regularization factor from current optimizer
     * 
     * @return float
     */
    public function getWeightDecay(): float
    {
        return $this->optimizer->getWeightDecay();
    }

    /**
     * Sets L1 regularization factor on current optimizer
     * 
     * @param float $l1
     */
    public function setL1Regularization(float $l1): void
    {
        $this->optimizer->setL1($l1);
    }

    /**
     * Gets L1 regularization factor from current optimizer
     * 
     * @return float
     */
    public function getL1Regularization(): float
    {
        return $this->optimizer->getL1();
    }

    /**
     * Resets optimizer and schedule states
     */
    public function resetOptimizer(): void
    {
        $this->optimizer->reset();
        $this->lrSchedule->reset();
    }

    /**
     * Sets maximum absolute value for gradient clipping (by value)
     * 
     * @param float|null $clipValue Positive number or null to disable
     */
    public function setClipValue(?float $clipValue): void
    {
        if ($clipValue !== null && ($clipValue <= 0.0 || !is_finite($clipValue))) {
            throw ValidationException::invalidValue('clipValue', $clipValue, 'positive finite float or null');
        }
        $this->clipValue = $clipValue;
    }

    public function getClipValue(): ?float
    {
        return $this->clipValue;
    }

    /**
     * Sets maximum Euclidean norm for gradient clipping (by norm)
     * 
     * @param float|null $clipNorm Positive number or null to disable
     */
    public function setClipNorm(?float $clipNorm): void
    {
        if ($clipNorm !== null && ($clipNorm <= 0.0 || !is_finite($clipNorm))) {
            throw ValidationException::invalidValue('clipNorm', $clipNorm, 'positive finite float or null');
        }
        $this->clipNorm = $clipNorm;
    }

    public function getClipNorm(): ?float
    {
        return $this->clipNorm;
    }

    /**
     * Applies gradient clipping to weight gradients array
     * 
     * @param array<float> &$gradients
     */
    private function applyGradientClipping(array &$gradients): void
    {
        if ($this->clipValue !== null && $this->clipValue > 0.0) {
            $c = $this->clipValue;
            for ($i = 0; $i < count($gradients); $i++) {
                if ($gradients[$i] > $c) {
                    $gradients[$i] = $c;
                } elseif ($gradients[$i] < -$c) {
                    $gradients[$i] = -$c;
                }
            }
        }

        if ($this->clipNorm !== null && $this->clipNorm > 0.0) {
            $sumSq = 0.0;
            foreach ($gradients as $g) {
                $sumSq += $g * $g;
            }
            $norm = sqrt($sumSq);
            if ($norm > $this->clipNorm && $norm > 1e-12) {
                $scale = $this->clipNorm / $norm;
                for ($i = 0; $i < count($gradients); $i++) {
                    $gradients[$i] *= $scale;
                }
            }
        }
    }

    /**
     * Returns the training loss history
     * 
     * @return array<float> Array of loss values from each training step
     * 
     * @example
     * ```php
     * $trainer->trainBatch($data, 50);
     * $losses = $trainer->getLossHistory();
     * echo "Loss decreased from " . $losses[0] . " to " . end($losses) . "\n";
     * ```
     */
    public function getLossHistory(): array
    {
        return $this->lossHistory;
    }

    /**
     * Returns the current epoch number
     * 
     * @return int Current epoch (starts from 0)
     * 
     * @example
     * ```php
     * echo "Currently at epoch: " . $trainer->getCurrentEpoch() . "\n";
     * ```
     */
    public function getCurrentEpoch(): int
    {
        return $this->currentEpoch;
    }

    /**
     * Clears the training history and resets epoch counter
     * 
     * @example
     * ```php
     * $trainer->clearHistory();
     * // Fresh start for new training session
     * ```
     */
    public function clearHistory(): void
    {
        $this->lossHistory = [];
        $this->currentEpoch = 0;
    }

    /**
     * Performs forward pass through the network with caching
     * 
     * Propagates input through all layers while caching intermediate
     * results needed for backpropagation. This includes weighted sums,
     * activations, and inputs for each layer.
     * 
     * @param array $input Input vector for the network
     * 
     * @return array Output vector from the final layer
     * 
     * @throws NetworkException If forward pass fails
     * @throws ValidationException If input is invalid
     * 
     * @example
     * ```php
     * $input = [0.1, 0.5, 0.8];
     * $output = $trainer->forwardPass($input);
     * // Network state is now cached for backward pass
     * ```
     */
    public function forwardPass(array $input): array
    {
        if (empty($input)) {
            throw ValidationException::invalidValue('input', 'empty array', 'non-empty array');
        }

        try {
            // Clear any existing cache first
            $this->clearNetworkCache();
            
            // Get network layers
            $numLayers = $this->network->GetNrLayers();
            if ($numLayers === 0) {
                throw NetworkException::invalidArchitecture('network has no layers');
            }

            $currentInput = $input;
            
            // Process each layer with caching
            for ($layerIndex = 0; $layerIndex < $numLayers; $layerIndex++) {
                $layer = $this->network->GetLayer($layerIndex);
                $layerOutput = [];
                $numCells = $layer->GetSize();
                
                if ($numCells === 0) {
                    throw NetworkException::invalidArchitecture("layer {$layerIndex} has no cells");
                }
                
                // Adjust input size to match layer's cell weight dimension
                $expectedSize = $layer->GetCell(0)->GetSize();
                if (count($currentInput) < $expectedSize) {
                    $currentInput = array_pad($currentInput, $expectedSize, 0.0);
                } elseif (count($currentInput) > $expectedSize) {
                    $currentInput = array_slice($currentInput, 0, $expectedSize);
                }
                
                // Process each cell in the layer
                for ($cellIndex = 0; $cellIndex < $numCells; $cellIndex++) {
                    try {
                        $cell = $layer->GetCell($cellIndex);
                        $cellOutput = $cell->ForwardPass($currentInput);
                        $layerOutput[] = $cellOutput;
                    } catch (\Exception $e) {
                        throw NetworkException::incompatibleOperation(
                            'cell forward pass',
                            "Failed forward pass for cell {$cellIndex} in layer {$layerIndex}: " . $e->getMessage()
                        );
                    }
                }
                
                $currentInput = $layerOutput;
            }
            
            return $currentInput;
            
        } catch (NetworkException $e) {
            throw $e;
        } catch (\Exception $e) {
            throw NetworkException::incompatibleOperation(
                'forward pass',
                "Forward pass failed: " . $e->getMessage()
            );
        }
    }

    /**
     * Performs backward pass to compute and apply gradients
     * 
     * Implements the backpropagation algorithm to compute gradients for all
     * network weights and applies them using gradient descent. This includes:
     * 1. Computing output layer gradients from loss derivative
     * 2. Propagating gradients backward through hidden layers
     * 3. Updating all weights using computed gradients
     * 
     * @param array $predicted Network output from forward pass
     * @param array $expected Target output values
     * 
     * @throws NetworkException If backward pass fails
     * @throws ComputationException If gradient computation fails
     * @throws ValidationException If inputs are invalid
     * 
     * @example
     * ```php
     * $predicted = $trainer->forwardPass($input);
     * $trainer->backwardPass($predicted, $expected);
     * // Network weights are now updated
     * ```
     */
    public function backwardPass(array $predicted, array $expected): void
    {
        if (empty($predicted)) {
            throw ValidationException::invalidValue('predicted', 'empty array', 'non-empty array');
        }
        
        if (empty($expected)) {
            throw ValidationException::invalidValue('expected', 'empty array', 'non-empty array');
        }
        
        if (count($predicted) !== count($expected)) {
            throw ValidationException::invalidArraySize(
                'predicted',
                count($predicted),
                count($expected)
            );
        }

        try {
            // Get network layers (we'll process them in reverse order)
            $numLayers = $this->network->GetNrLayers();
            
            if ($numLayers === 0) {
                throw NetworkException::invalidArchitecture('network has no layers');
            }

            // Compute loss gradient (derivative of loss w.r.t. predicted output)
            $lossGradient = $this->lossFunction->derivative($predicted, $expected);
            
            if (count($lossGradient) !== count($predicted)) {
                throw ComputationException::invalidNumericalValue(
                    count($lossGradient),
                    "Loss gradient size mismatch: got " . count($lossGradient) . 
                    ", expected " . count($predicted)
                );
            }

            // Start backward propagation from output layer
            $currentGradient = $lossGradient;
            
            // Process layers in reverse order (output to input)
            for ($layerIndex = $numLayers - 1; $layerIndex >= 0; $layerIndex--) {
                $layer = $this->network->GetLayer($layerIndex);
                $numCells = $layer->GetSize();
                
                if ($numCells === 0) {
                    throw NetworkException::invalidArchitecture("layer {$layerIndex} has no cells");
                }
                
                // Ensure gradient size matches layer size
                if (count($currentGradient) !== $numCells) {
                    throw ComputationException::invalidNumericalValue(
                        count($currentGradient),
                        "Gradient size mismatch for layer {$layerIndex}: got " . 
                        count($currentGradient) . ", expected " . $numCells
                    );
                }
                
                $nextGradient = null;
                
                // Process each cell in the current layer
                for ($cellIndex = 0; $cellIndex < $numCells; $cellIndex++) {
                    try {
                        $cell = $layer->GetCell($cellIndex);
                        
                        // Get the error signal for this cell
                        $cellErrorSignal = $currentGradient[$cellIndex];
                        
                        // Compute activation derivative for this cell
                        $activationDerivative = $cell->getActivationDerivative();
                        
                        // Local gradient = error signal * activation derivative
                        $localGradient = $cellErrorSignal * $activationDerivative;
                        
                        // Get cached inputs for weight gradient computation
                        $cellInputs = $cell->getLastInputs();
                        
                        // Compute weight gradients: gradient = local_gradient * input
                        $weightGradients = [];
                        foreach ($cellInputs as $input) {
                            $weightGradients[] = $localGradient * $input;
                        }
                        $this->applyGradientClipping($weightGradients);
                        
                        // For hidden layers, compute gradients for previous layer BEFORE updating weights
                        if ($layerIndex > 0) {
                            $cellWeights = $cell->GetDna();
                            
                            // Initialize next gradient array if needed
                            if ($nextGradient === null) {
                                $nextGradient = array_fill(0, count($cellWeights), 0.0);
                            }
                            
                            // Accumulate gradients for previous layer
                            // gradient[i] += weight[i] * local_gradient
                            for ($i = 0; $i < count($cellWeights); $i++) {
                                if (!isset($nextGradient[$i])) {
                                    $nextGradient[$i] = 0.0;
                                }
                                $nextGradient[$i] += $cellWeights[$i] * $localGradient;
                            }
                        }
                        
                        // Apply weight updates via optimizer
                        $this->optimizer->update($cell, $weightGradients, $this->learningRate);
                        
                    } catch (\Exception $e) {
                        throw NetworkException::incompatibleOperation(
                            'cell backward pass',
                            "Failed backward pass for cell {$cellIndex} in layer {$layerIndex}: " . $e->getMessage()
                        );
                    }
                }
                
                // Move to previous layer
                $currentGradient = $nextGradient;
            }
            
        } catch (NetworkException $e) {
            throw $e;
        } catch (ComputationException $e) {
            throw $e;
        } catch (\Exception $e) {
            throw NetworkException::incompatibleOperation(
                'backward pass',
                "Backward pass failed: " . $e->getMessage()
            );
        }
    }

    /**
     * Computes gradients for all weights in the network without updating them.
     * Useful for gradient inspection, mini-batch accumulation, and verification.
     * 
     * @param array $input Input vector
     * @param array $expected Expected output vector
     * 
     * @return array<int, array<int, array<float>>> Gradients indexed by [layerIndex][cellIndex]
     */
    public function computeGradients(array $input, array $expected): array
    {
        $predicted = $this->forwardPass($input);

        $numLayers = $this->network->GetNrLayers();
        if ($numLayers === 0) {
            throw NetworkException::invalidArchitecture('network has no layers');
        }

        $lossGradient = $this->lossFunction->derivative($predicted, $expected);
        $currentGradient = $lossGradient;
        $allGradients = [];

        for ($layerIndex = $numLayers - 1; $layerIndex >= 0; $layerIndex--) {
            $layer = $this->network->GetLayer($layerIndex);
            $numCells = $layer->GetSize();
            $nextGradient = null;
            $allGradients[$layerIndex] = [];

            for ($cellIndex = 0; $cellIndex < $numCells; $cellIndex++) {
                $cell = $layer->GetCell($cellIndex);
                $cellErrorSignal = $currentGradient[$cellIndex];
                $activationDerivative = $cell->getActivationDerivative();
                $localGradient = $cellErrorSignal * $activationDerivative;
                $cellInputs = $cell->getLastInputs();

                $weightGradients = [];
                foreach ($cellInputs as $val) {
                    $weightGradients[] = $localGradient * $val;
                }
                $this->applyGradientClipping($weightGradients);
                $allGradients[$layerIndex][$cellIndex] = $weightGradients;

                if ($layerIndex > 0) {
                    $cellWeights = $cell->GetDna();
                    if ($nextGradient === null) {
                        $nextGradient = array_fill(0, count($cellWeights), 0.0);
                    }
                    for ($i = 0; $i < count($cellWeights); $i++) {
                        if (!isset($nextGradient[$i])) {
                            $nextGradient[$i] = 0.0;
                        }
                        $nextGradient[$i] += $cellWeights[$i] * $localGradient;
                    }
                }
            }
            $currentGradient = $nextGradient;
        }

        return $allGradients;
    }

    /**
     * Trains the network on a single input-output example
     * 
     * Performs one complete training step including forward pass,
     * loss computation, backward pass, and weight updates.
     * 
     * @param array $input Input vector
     * @param array $expected Expected output vector
     * 
     * @return float Computed loss for this training example
     * 
     * @throws TrainingException If training step fails
     * 
     * @example
     * ```php
     * $input = [0.2, 0.8];
     * $expected = [0.1, 0.9];
     * $loss = $trainer->trainSingle($input, $expected);
     * echo "Training loss: {$loss}\n";
     * ```
     */
    public function trainSingle(array $input, array $expected): float
    {
        if (empty($input)) {
            throw TrainingException::invalidTrainingData('input array cannot be empty');
        }
        
        if (empty($expected)) {
            throw TrainingException::invalidTrainingData('expected array cannot be empty');
        }

        try {
            // Forward pass
            $predicted = $this->forwardPass($input);
            
            // Ensure output and expected have same size
            if (count($predicted) !== count($expected)) {
                // Adjust sizes to match
                $targetSize = count($expected);
                if (count($predicted) < $targetSize) {
                    $predicted = array_pad($predicted, $targetSize, 0.0);
                } else {
                    $predicted = array_slice($predicted, 0, $targetSize);
                }
            }
            
            // Compute loss
            $loss = $this->lossFunction->calculate($predicted, $expected);
            
            if (!is_finite($loss)) {
                throw ComputationException::invalidNumericalValue(
                    $loss,
                    'computed loss must be finite'
                );
            }
            
            // Backward pass and weight updates
            $this->backwardPass($predicted, $expected);
            
            // Record loss
            $this->lossHistory[] = $loss;
            
            return $loss;
            
        } catch (NetworkException $e) {
            throw TrainingException::convergenceError(
                $this->currentEpoch,
                isset($loss) ? $loss : INF,
                "Training failed: " . $e->getMessage()
            );
        } catch (ComputationException $e) {
            throw TrainingException::convergenceError(
                $this->currentEpoch,
                isset($loss) ? $loss : INF,
                "Computation error during training: " . $e->getMessage()
            );
        } catch (\Exception $e) {
            throw TrainingException::invalidTrainingData(
                "Training step failed: " . $e->getMessage(),
                ['input' => $input, 'expected' => $expected]
            );
        }
    }

    /**
     * Trains the network on a batch of training examples
     * 
     * Performs multiple epochs of training on the provided dataset.
     * Each epoch processes all training examples once.
     * 
     * @param array $trainingData Array of [input, expected] pairs
     * @param int $epochs Number of training epochs (1-10000)
     * @param float|null $targetLoss Optional early stopping loss threshold
     * 
     * @return array<float> Loss values for each epoch
     * 
     * @throws TrainingException If batch training fails
     * @throws ValidationException If parameters are invalid
     * 
     * @example
     * ```php
     * $data = [
     *     [[0.0, 0.0], [0.0]],  // XOR training data
     *     [[0.0, 1.0], [1.0]],
     *     [[1.0, 0.0], [1.0]],
     *     [[1.0, 1.0], [0.0]]
     * ];
     * 
     * $epochLosses = $trainer->trainBatch($data, 1000, 0.01);
     * echo "Training completed. Final loss: " . end($epochLosses) . "\n";
     * ```
     */
    /**
     * Trains the network on a batch of training examples
     * 
     * Performs multiple epochs of training on the provided dataset.
     * Each epoch processes all training examples once.
     * 
     * @param array $trainingData Array of [input, expected] pairs
     * @param int $epochs Number of training epochs (1-10000)
     * @param float|null $targetLoss Optional early stopping loss threshold
     * @param int $batchSize Size of mini-batch for gradient updates (1 = online SGD)
     * 
     * @return array<float> Loss values for each epoch
     * 
     * @throws TrainingException If batch training fails
     * @throws ValidationException If parameters are invalid
     * 
     * @example
     * ```php
     * $data = [
     *     [[0.0, 0.0], [0.0]],  // XOR training data
     *     [[0.0, 1.0], [1.0]],
     *     [[1.0, 0.0], [1.0]],
     *     [[1.0, 1.0], [0.0]]
     * ];
     * 
     * $epochLosses = $trainer->trainBatch($data, 1000, 0.01);
     * echo "Training completed. Final loss: " . end($epochLosses) . "\n";
     * ```
     */
    public function trainBatch(array $trainingData, int $epochs, ?float $targetLoss = null, int $batchSize = 1): array
    {
        if (empty($trainingData)) {
            throw TrainingException::invalidTrainingData('training data cannot be empty');
        }
        
        if ($epochs <= 0) {
            throw ValidationException::outOfRange('epochs', $epochs, 1, 10000);
        }
        
        if ($epochs > 10000) {
            throw ValidationException::outOfRange('epochs', $epochs, 1, 10000);
        }
        
        if ($batchSize <= 0) {
            throw ValidationException::invalidValue('batchSize', $batchSize, 'positive integer >= 1');
        }

        if ($targetLoss !== null && (!is_numeric($targetLoss) || $targetLoss < 0)) {
            throw ValidationException::invalidValue(
                'targetLoss',
                $targetLoss,
                'positive number or null'
            );
        }

        // Validate training data format
        foreach ($trainingData as $index => $example) {
            if (!is_array($example) || count($example) !== 2) {
                throw TrainingException::invalidTrainingData(
                    "Training example {$index} must be [input, expected] array"
                );
            }
            
            if (!is_array($example[0]) || !is_array($example[1])) {
                throw TrainingException::invalidTrainingData(
                    "Training example {$index}: both input and expected must be arrays"
                );
            }
            
            if (empty($example[0]) || empty($example[1])) {
                throw TrainingException::invalidTrainingData(
                    "Training example {$index}: input and expected arrays cannot be empty"
                );
            }
        }

        $epochLosses = [];
        
        try {
            for ($epoch = 0; $epoch < $epochs; $epoch++) {
                $this->currentEpoch = $epoch;
                
                // Adjust learning rate via schedule at the start of epoch
                $lastLoss = empty($epochLosses) ? null : end($epochLosses);
                $this->learningRate = $this->lrSchedule->getLearningRate($epoch, $this->initialLearningRate, $lastLoss);

                $epochTotalLoss = 0.0;
                $exampleCount = 0;
                
                if ($batchSize <= 1) {
                    // Online stochastic gradient descent (update per example)
                    foreach ($trainingData as $index => $example) {
                        try {
                            $input = $example[0];
                            $expected = $example[1];
                            
                            $loss = $this->trainSingle($input, $expected);
                            $epochTotalLoss += $loss;
                            $exampleCount++;
                            
                        } catch (\Exception $e) {
                            throw TrainingException::convergenceError(
                                $epoch,
                                $epochTotalLoss / max(1, $exampleCount),
                                "Failed to train on example {$index}: " . $e->getMessage()
                            );
                        }
                    }
                } else {
                    // Mini-batch gradient accumulation
                    $chunks = array_chunk($trainingData, $batchSize);
                    foreach ($chunks as $chunk) {
                        $accumulatedGradients = [];
                        $batchCount = count($chunk);

                        foreach ($chunk as $index => $example) {
                            $input = $example[0];
                            $expected = $example[1];

                            $predicted = $this->forwardPass($input);
                            $loss = $this->lossFunction->calculate($predicted, $expected);
                            $epochTotalLoss += $loss;
                            $exampleCount++;

                            $grads = $this->computeGradients($input, $expected);
                            foreach ($grads as $layerIdx => $layerGrads) {
                                if (!isset($accumulatedGradients[$layerIdx])) {
                                    $accumulatedGradients[$layerIdx] = [];
                                }
                                foreach ($layerGrads as $cellIdx => $cellGrads) {
                                    if (!isset($accumulatedGradients[$layerIdx][$cellIdx])) {
                                        $accumulatedGradients[$layerIdx][$cellIdx] = array_fill(0, count($cellGrads), 0.0);
                                    }
                                    for ($w = 0; $w < count($cellGrads); $w++) {
                                        $accumulatedGradients[$layerIdx][$cellIdx][$w] += $cellGrads[$w];
                                    }
                                }
                            }
                        }

                        // Apply averaged gradients for the mini-batch
                        $numLayers = $this->network->GetNrLayers();
                        for ($l = 0; $l < $numLayers; $l++) {
                            $layer = $this->network->GetLayer($l);
                            $numCells = $layer->GetSize();
                            for ($c = 0; $c < $numCells; $c++) {
                                if (isset($accumulatedGradients[$l][$c])) {
                                    $avgGrads = array_map(function($g) use ($batchCount) {
                                        return $g / $batchCount;
                                    }, $accumulatedGradients[$l][$c]);
                                    $this->applyGradientClipping($avgGrads);
                                    $this->optimizer->update($layer->GetCell($c), $avgGrads, $this->learningRate);
                                }
                            }
                        }
                    }
                }
                
                // Compute average loss for this epoch
                $avgLoss = $epochTotalLoss / $exampleCount;
                $epochLosses[] = $avgLoss;
                
                // Check for early stopping
                if ($targetLoss !== null && $avgLoss <= $targetLoss) {
                    break;
                }
                
                // Check for convergence problems
                if (!is_finite($avgLoss)) {
                    throw TrainingException::convergenceError(
                        $epoch,
                        $avgLoss,
                        "Loss became non-finite during training"
                    );
                }
            }
            
            return $epochLosses;
            
        } catch (TrainingException $e) {
            throw $e;
        } catch (\Exception $e) {
            throw TrainingException::invalidTrainingData(
                "Batch training failed: " . $e->getMessage(),
                ['epochs' => $epochs, 'data_size' => count($trainingData)]
            );
        }
    }

    /**
     * Clears cached intermediate results from all network cells
     * 
     * @throws NetworkException If cache clearing fails
     */
    private function clearNetworkCache(): void
    {
        try {
            $numLayers = $this->network->GetNrLayers();
            
            for ($layerIndex = 0; $layerIndex < $numLayers; $layerIndex++) {
                $layer = $this->network->GetLayer($layerIndex);
                $numCells = $layer->GetSize();
                
                for ($cellIndex = 0; $cellIndex < $numCells; $cellIndex++) {
                    $cell = $layer->GetCell($cellIndex);
                    $cell->clearCache();
                }
            }
        } catch (\Exception $e) {
            throw NetworkException::incompatibleOperation(
                'cache clearing',
                "Failed to clear network cache: " . $e->getMessage()
            );
        }
    }
}