<?php

/**
 * Multi-Class Digit Recognition (0-9) - Training Script
 *
 * Trains a heterogeneous 2-layer neural network:
 *   Input (35 pixels, 5x7) -> Hidden (24 Tanh neurons) -> Output (10 Sigmoid neurons)
 * using Backpropagation with Adam optimizer and StepDecay learning-rate schedule.
 *
 * Usage:
 *   php examples/digit_recognition/train_digits.php
 */

require_once __DIR__ . '/common.php';

use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\schedule\StepDecaySchedule;

echo "\n============================================================\n";
echo "   🔢 NiNa Multi-Class Digit Recognition (0-9) - Trainer\n";
echo "============================================================\n\n";

try {
    $startTime = microtime(true);

    // 1. Export canonical 5x7 pattern files if not present
    $patternsDir = __DIR__ . DIRECTORY_SEPARATOR . 'patterns';
    if (!is_dir($patternsDir)) {
        mkdir($patternsDir, 0755, true);
    }

    $baseDigits = getBaseDigitPatterns();
    foreach ($baseDigits as $digit => $bitmap) {
        $rows = [];
        for ($r = 0; $r < DIGIT_HEIGHT; $r++) {
            $rows[] = implode('', array_slice($bitmap, $r * DIGIT_WIDTH, DIGIT_WIDTH));
        }
        file_put_contents(
            $patternsDir . DIRECTORY_SEPARATOR . "digit_{$digit}.txt",
            implode("\n", $rows) . "\n"
        );
    }

    // 2. Build network & dataset
    $hiddenSize = 24;
    $samplesPerDigit = 8;
    $trainNoise = 0.08;
    $maxEpochs = 60;
    $learningRate = 0.025;
    $targetLoss = 0.003;

    $network = createDigitNetwork($hiddenSize);
    $trainDataset = generateDigitDataset($samplesPerDigit, $trainNoise);
    $trainPairs = toBackpropPairs($trainDataset);

    echo "📋 Network Architecture (Heterogeneous):\n";
    echo "   • Input size:  " . DIGIT_INPUT_SIZE . " pixels (5×7 bitmap)\n";
    echo "   • Hidden layer: {$hiddenSize} neurons (Tanh, Xavier initialized)\n";
    echo "   • Output layer: " . DIGIT_CLASSES . " neurons (Sigmoid, one-hot 0–9)\n\n";

    echo "📊 Dataset & Training Configuration:\n";
    echo "   • Training samples: " . count($trainDataset) . " ({$samplesPerDigit} per digit, canonical + LCD variants + {$trainNoise} noise)\n";
    echo "   • Optimizer:        Adam (lr={$learningRate}, StepDecay every 20 epochs)\n";
    echo "   • Loss function:    Mean Squared Error (target ≤ {$targetLoss})\n";
    echo "   • Max epochs:       {$maxEpochs}\n\n";

    $initialEval = evaluateDigitDataset($network, $trainDataset);
    echo sprintf("🚀 Starting training (initial accuracy: %.1f%%)...\n", $initialEval['accuracy'] * 100);

    // 3. Configure BackpropagationTrainer
    $optimizer = new Adam();
    $schedule = new StepDecaySchedule(20, 0.5);
    $trainer = new BackpropagationTrainer(
        $network,
        new MeanSquaredError(),
        $learningRate,
        $optimizer,
        $schedule
    );

    $epochLosses = $trainer->trainBatch($trainPairs, $maxEpochs, $targetLoss, 1);
    $trainedNetwork = $trainer->getTrainedNetwork();

    $elapsed = microtime(true) - $startTime;
    $epochsRun = count($epochLosses);
    $finalLoss = end($epochLosses);

    // 4. Evaluate on training set and unseen noisy test set
    $trainEval = evaluateDigitDataset($trainedNetwork, $trainDataset);

    $noisyTestSet = [];
    foreach ($baseDigits as $digit => $pattern) {
        $noisyTestSet[] = [
            'input' => addDigitNoise($pattern, 0.15),
            'target' => createOneHotTarget($digit),
            'digit' => $digit,
        ];
    }
    $noisyEval = evaluateDigitDataset($trainedNetwork, $noisyTestSet);

    echo "\n✅ Training Complete!\n";
    echo "============================================================\n";
    echo sprintf("   • Epochs used:       %d / %d\n", $epochsRun, $maxEpochs);
    echo sprintf("   • Initial → Final MSE: %.6f → %.6f\n", $epochLosses[0], $finalLoss);
    echo sprintf("   • Training accuracy: %d/%d (%.1f%%)\n", $trainEval['correct'], $trainEval['total'], $trainEval['accuracy'] * 100);
    echo sprintf("   • Noisy test (15%%):  %d/%d (%.1f%%)\n", $noisyEval['correct'], $noisyEval['total'], $noisyEval['accuracy'] * 100);
    echo sprintf("   • Training time:     %.2f sec\n\n", $elapsed);

    // 5. Per-digit evaluation table on canonical digits 0-9
    echo "🔍 Per-Digit Classification Results (Digits 0–9):\n";
    echo "------------------------------------------------------------\n";
    echo " Digit | Predicted | Confidence | Second Choice | Status\n";
    echo "-------+-----------+------------+---------------+-----------\n";

    foreach ($baseDigits as $digit => $pattern) {
        $pred = predictDigit($trainedNetwork, $pattern);
        $keys = array_keys($pred['rankings']);
        $top1 = $keys[0];
        $top2 = $keys[1];
        $conf1 = $pred['rankings'][$top1] * 100;
        $conf2 = $pred['rankings'][$top2] * 100;
        $ok = ($top1 === $digit) ? '✅ PASS' : '❌ FAIL';

        echo sprintf(
            "   %d   |     %d     |   %5.1f%%   |   %d (%4.1f%%)   | %s\n",
            $digit,
            $top1,
            $conf1,
            $top2,
            $conf2,
            $ok
        );
    }
    echo "------------------------------------------------------------\n\n";

    // 6. Save model
    $modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'digit_model.json';
    file_put_contents($modelPath, $trainedNetwork->Json());
    echo "💾 Trained model saved to: {$modelPath}\n";
    echo "📁 Pattern files saved in: {$patternsDir}\n\n";
    echo "Run inference with:\n";
    echo "  php examples/digit_recognition/recognize_digit.php\n";
    echo "  php examples/digit_recognition/recognize_digit.php examples/digit_recognition/patterns/digit_8.txt\n";
    echo "  php examples/digit_recognition/recognize_digit.php --digit 3 --noise 0.15\n\n";

} catch (\Throwable $e) {
    fwrite(STDERR, "❌ Error: " . $e->getMessage() . "\n");
    exit(1);
}
