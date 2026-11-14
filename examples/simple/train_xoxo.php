<?php

/**
 * XOXO (Tic-Tac-Toe) Neural Network Trainer
 * 
 * Trains a neural network to evaluate Tic-Tac-Toe board states and select optimal moves.
 * 
 * Supports two training paradigms:
 *   - Backpropagation (Default): Fast gradient descent with Adam optimizer (reaches 100% accuracy)
 *   - Genetic Algorithm (--ga): Evolutionary population-based training
 * 
 * Usage:
 *   php examples/simple/train_xoxo.php            # Train using Backpropagation
 *   php examples/simple/train_xoxo.php --ga       # Train using Genetic Algorithm
 */

require_once __DIR__ . "/../../src/autoload.php";

use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\Trainer;
use mc\nina\Predictor;
use mc\nina\activation\Sigmoid;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\initializer\XavierInitializer;

$useGA = in_array('--ga', $argv, true) || in_array('--genetic', $argv, true);

// 12 Canonical Tic-Tac-Toe Training Samples
// 0 = Empty, 1 = X (Player), 2 = O (Opponent)
$TRAIN_DATA = [
    [
        'input'  => [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'target' => [1, 0, 0, 0, 0, 0, 0, 0, 0] // Opening: take top-left corner
    ],
    [
        'input'  => [1, 2, 0, 0, 0, 0, 0, 0, 0],
        'target' => [1, 2, 0, 1, 0, 0, 0, 0, 0] // Take middle-left
    ],
    [
        'input'  => [1, 0, 2, 0, 0, 0, 0, 0, 0],
        'target' => [1, 0, 2, 1, 0, 0, 0, 0, 0] // Take middle-left
    ],
    [
        'input'  => [1, 0, 0, 2, 0, 0, 0, 0, 0],
        'target' => [1, 1, 0, 2, 0, 0, 0, 0, 0] // Take top-center
    ],
    [
        'input'  => [1, 0, 0, 0, 2, 0, 0, 0, 0],
        'target' => [1, 0, 0, 1, 2, 0, 0, 0, 0] // Respond to center
    ],
    [
        'input'  => [1, 0, 0, 0, 0, 2, 0, 0, 0],
        'target' => [1, 0, 1, 0, 0, 2, 0, 0, 0] // Take top-right
    ],
    [
        'input'  => [1, 0, 0, 0, 0, 0, 2, 0, 0],
        'target' => [1, 1, 0, 0, 0, 0, 2, 0, 0] // Take top-center
    ],
    [
        'input'  => [1, 0, 0, 0, 0, 0, 0, 2, 0],
        'target' => [1, 0, 1, 0, 0, 0, 0, 2, 0] // Take top-right
    ],
    [
        'input'  => [1, 0, 0, 0, 0, 0, 0, 0, 2],
        'target' => [1, 0, 0, 0, 0, 0, 1, 0, 2] // Take bottom-left
    ],
    [
        'input'  => [1, 2, 0, 1, 0, 0, 2, 0, 0],
        'target' => [1, 2, 0, 1, 1, 0, 2, 0, 0] // Take center
    ],
    [
        'input'  => [1, 2, 0, 1, 1, 0, 2, 0, 2],
        'target' => [1, 2, 0, 1, 1, 1, 2, 0, 2] // Complete row (winning move)
    ],
    [
        'input'  => [1, 2, 0, 1, 1, 0, 2, 2, 0],
        'target' => [1, 2, 0, 1, 1, 0, 2, 2, 1] // Complete column (winning move)
    ],
    [
        'input'  => [0, 0, 0, 0, 0, 0, 1, 0, 0],
        'target' => [0, 0, 0, 0, 2, 0, 1, 0, 0] // O response: take center against corner opening
    ],
    [
        'input'  => [0, 0, 0, 0, 0, 2, 1, 0, 0],
        'target' => [0, 0, 0, 0, 1, 2, 1, 0, 0] // X response: seize center when O plays edge
    ],
];

// Network architecture: 3 layers x 9 neurons
$NET_NR_LAYERS = 3;
$NET_LAYER_SIZE = 9;
$NET_FILE = __DIR__ . '/xoxo_model.json';

echo "\n=========================================\n";
echo "      XOXO Neural Network Trainer        \n";
echo "=========================================\n";
echo "Training Samples: " . count($TRAIN_DATA) . "\n";
echo "Architecture:     {$NET_NR_LAYERS} layers × {$NET_LAYER_SIZE} neurons (Sigmoid)\n";
echo "Method:           " . ($useGA ? "Genetic Algorithm (Evolutionary)" : "Backpropagation (Gradient Descent / Adam)") . "\n";
echo "Output Model:     {$NET_FILE}\n\n";

try {
    // 1. Initialize or Load Network
    $resume = in_array('--resume', $argv, true);
    if ($resume && file_exists($NET_FILE)) {
        echo "Resuming training from existing model: {$NET_FILE}...\n";
        $network = Network::Load(file_get_contents($NET_FILE));
    } else {
        echo "Creating fresh neural network with Xavier initialization...\n";
        $network = Network::Create("XOXO_AI", $NET_NR_LAYERS, $NET_LAYER_SIZE, new Sigmoid());
        $network->initializeWeights(new XavierInitializer());
    }

    $startTime = microtime(true);

    if (!$useGA) {
        // =========================================================================
        // BACKPROPAGATION TRAINING (Fast, Accurate, Recommended)
        // =========================================================================
        $learningRate = 0.05;
        $maxEpochs = 500;
        $targetLoss = 0.005;

        echo "Configuring Backpropagation Trainer (Adam, lr={$learningRate})...\n";
        $trainer = new BackpropagationTrainer(
            $network,
            new MeanSquaredError(),
            $learningRate,
            new Adam()
        );

        // Normalize dataset:
        // Input: 0.0 = empty, 0.5 = X, 1.0 = O
        // Target: 1.0 for the expected move position, 0.0 for other cells
        $bpDataset = [];
        foreach ($TRAIN_DATA as $sample) {
            $normInput = array_map(fn($v) => $v / 2.0, $sample['input']);
            $targetVector = [];
            for ($i = 0; $i < 9; $i++) {
                $targetVector[] = ($sample['input'][$i] === 0 && $sample['target'][$i] !== 0) ? 1.0 : 0.0;
            }
            $bpDataset[] = [$normInput, $targetVector];
        }

        echo "Training for up to {$maxEpochs} epochs (target loss: {$targetLoss})...\n";
        $epochLosses = $trainer->trainBatch($bpDataset, $maxEpochs, $targetLoss);

        $trainingTime = microtime(true) - $startTime;
        $finalLoss = end($epochLosses);
        $totalEpochs = count($epochLosses);

        echo "✓ Training finished in " . round($trainingTime, 3) . "s ({$totalEpochs} epochs, final loss: " . round($finalLoss, 6) . ")\n\n";
        $bestNetwork = $trainer->getNetwork();

    } else {
        // =========================================================================
        // GENETIC ALGORITHM TRAINING (Evolutionary)
        // =========================================================================
        $populationSize = 100;
        $mutationRate = 10;
        $maxGenerations = 1000;
        $targetAccuracy = 1.0;

        echo "Configuring Genetic Algorithm Trainer (pop={$populationSize}, mut={$mutationRate}%)...\n";
        $trainer = new Trainer($network, $populationSize, $mutationRate);

        // Encoding for GA string interface:
        // Map: 0 -> chr(0) (0.0), 1 -> chr(127) (~0.5), 2 -> chr(255) (1.0)
        $gaData = [];
        foreach ($TRAIN_DATA as $sample) {
            $inStr = implode('', array_map(fn($v) => chr((int)round($v * 127.5)), $sample['input']));
            $targetStr = implode('', array_map(fn($v) => chr((int)round($v * 127.5)), $sample['target']));
            $gaData[] = ['input' => $inStr, 'target' => $targetStr, 'raw' => $sample];
        }

        echo "Starting evolutionary training (max {$maxGenerations} generations)...\n";
        $bestAcc = 0.0;

        for ($gen = 1; $gen <= $maxGenerations; $gen++) {
            // Train on all samples in this generation
            foreach ($gaData as $item) {
                $trainer->Train($item['input'], $item['target']);
            }

            // Periodically evaluate move accuracy
            if ($gen % 25 === 0 || $gen === 1 || $gen === $maxGenerations) {
                $candidate = $trainer->getBestNetwork();
                $correct = evaluateMoveAccuracy($candidate, $TRAIN_DATA);
                $acc = $correct / count($TRAIN_DATA);

                if ($acc > $bestAcc) {
                    $bestAcc = $acc;
                }

                echo "Generation {$gen}: accuracy = " . round($acc * 100, 1) . "% (best: " . round($bestAcc * 100, 1) . "%)\n";

                if ($acc >= $targetAccuracy) {
                    echo "🎯 Target accuracy reached at generation {$gen}!\n";
                    break;
                }
            }
        }

        $trainingTime = microtime(true) - $startTime;
        echo "✓ Evolution finished in " . round($trainingTime, 3) . "s\n\n";
        $bestNetwork = $trainer->getBestNetwork();
    }

    // =========================================================================
    // FINAL EVALUATION
    // =========================================================================
    echo "=========================================\n";
    echo "          Final Move Evaluation          \n";
    echo "=========================================\n";

    $correctMoves = 0;
    foreach ($TRAIN_DATA as $idx => $sample) {
        $normInput = array_map(fn($v) => $v / 2.0, $sample['input']);
        $scores = $bestNetwork->Calculate($normInput);

        // Find top predicted move among empty cells
        $bestPos = -1;
        $bestScore = -INF;
        for ($i = 0; $i < 9; $i++) {
            if ($sample['input'][$i] === 0 && $scores[$i] > $bestScore) {
                $bestScore = $scores[$i];
                $bestPos = $i;
            }
        }

        // Expected move position
        $expectedPos = -1;
        for ($i = 0; $i < 9; $i++) {
            if ($sample['input'][$i] === 0 && $sample['target'][$i] !== 0) {
                $expectedPos = $i;
                break;
            }
        }

        $isMatch = ($bestPos === $expectedPos);
        if ($isMatch) {
            $correctMoves++;
        }

        $status = $isMatch ? "✓ PASS" : "✗ FAIL";
        echo sprintf("Sample %2d: Expected pos %d, Got pos %d  [%s]\n", $idx + 1, $expectedPos + 1, $bestPos + 1, $status);
    }

    $finalAccuracy = ($correctMoves / count($TRAIN_DATA)) * 100;
    echo "\n🎯 Move Accuracy: " . round($finalAccuracy, 1) . "% ({$correctMoves}/" . count($TRAIN_DATA) . ")\n\n";

    // 3. Save Trained Model
    $json = $bestNetwork->Json();
    if (file_put_contents($NET_FILE, $json) !== false) {
        echo "💾 Model successfully saved to: {$NET_FILE}\n";
        echo "📊 Size: " . number_format(strlen($json)) . " bytes\n";
    } else {
        fwrite(STDERR, "⚠️  Failed to save model to {$NET_FILE}\n");
    }

    echo "\n🎉 Model is ready! You can test it by running:\n";
    echo "   php examples/simple/xoxo.php\n\n";

} catch (Exception $e) {
    fwrite(STDERR, "Error: " . $e->getMessage() . "\n");
    fwrite(STDERR, "File: " . $e->getFile() . ":" . $e->getLine() . "\n");
    exit(1);
}

/**
 * Evaluates move accuracy for a network against training data
 */
function evaluateMoveAccuracy(Network $network, array $trainData): int
{
    $correct = 0;
    foreach ($trainData as $sample) {
        $normInput = array_map(fn($v) => $v / 2.0, $sample['input']);
        $scores = $network->Calculate($normInput);

        $bestPos = -1;
        $bestScore = -INF;
        for ($i = 0; $i < 9; $i++) {
            if ($sample['input'][$i] === 0 && $scores[$i] > $bestScore) {
                $bestScore = $scores[$i];
                $bestPos = $i;
            }
        }

        $expectedPos = -1;
        for ($i = 0; $i < 9; $i++) {
            if ($sample['input'][$i] === 0 && $sample['target'][$i] !== 0) {
                $expectedPos = $i;
                break;
            }
        }

        if ($bestPos === $expectedPos) {
            $correct++;
        }
    }
    return $correct;
}
