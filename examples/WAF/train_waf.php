<?php

/**
 * NiNa Web Application Firewall (WAF) - Training & Benchmark Script
 *
 * Trains the 24 -> 32 -> 16 -> 5 baseline neural network (1,360 weights)
 * to classify HTTP payloads into 5 threat categories:
 *   [0] BENIGN, [1] SQLI, [2] XSS, [3] LFI, [4] RCE
 *
 * Usage:
 *   php examples/WAF/train_waf.php
 */

require_once __DIR__ . '/waf_common.php';

use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\schedule\PlateauSchedule;

echo "\n======================================================================\n";
echo "   🛡️  NiNa Neural WAF — Threat Classifier Training (24 → 32 → 16 → 5)\n";
echo "======================================================================\n\n";

try {
    $startTime = microtime(true);

    // 1. Create baseline 24 -> 32 -> 16 -> 5 network
    $network = createWafNetwork([32, 16]);
    $archStats = getNetworkArchitectureStats($network);

    echo "🏗️  Network Architecture Breakdown:\n";
    echo "   • Input Dimension: " . WAF_INPUT_SIZE . " normalized features (statistical + lexical + bias)\n";
    foreach ($archStats['layer_breakdown'] as $idx => $layerInfo) {
        echo sprintf(
            "   • Layer %d (%-14s): %2d neurons × %2d inputs = %4d weights (%s)\n",
            $idx + 1,
            $layerInfo['name'],
            $layerInfo['neurons'],
            $layerInfo['inputs_per_neuron'],
            $layerInfo['weights'],
            strtoupper($layerInfo['activation'])
        );
    }
    echo sprintf(
        "   • Total Capacity:  %d neurons, %s trainable weights\n\n",
        $archStats['total_neurons'],
        number_format($archStats['total_weights'])
    );

    // 2. Prepare dataset
    $corpus = getWafCorpus();
    $trainDataset = $corpus['train'];
    $trainUnique = $corpus['train_unique'];
    $testDataset = $corpus['test'];

    $trainPairs = [];
    foreach ($trainDataset as $sample) {
        $trainPairs[] = [$sample['input'], $sample['target']];
    }

    echo "📊 Dataset & Optimization Setup:\n";
    echo "   • Training samples:   " . count($trainDataset) . " (" . count($trainUnique) . " unique + " . count($trainUnique) . " jitter-augmented across 5 classes)\n";
    echo "   • Unseen test set:    " . count($testDataset) . " payloads (5 per class)\n";
    echo "   • Optimizer:          Adam (lr=0.02, L2 decay=0.0001, clipNorm=5.0)\n";
    echo "   • LR Schedule:        PlateauSchedule (factor=0.5, patience=5)\n\n";

    // 3. Configure BackpropagationTrainer with Adam, L2 weight decay, gradient clipping, and PlateauSchedule
    $optimizer = new Adam();
    $schedule = new PlateauSchedule(0.5, 5, 1e-4, 0.0005);
    $trainer = new BackpropagationTrainer(
        $network,
        new MeanSquaredError(),
        0.02,
        $optimizer,
        $schedule
    );
    $trainer->setWeightDecay(0.0001);
    $trainer->setClipNorm(5.0);

    $maxEpochs = 60;
    $targetLoss = 0.0025;

    echo "🚀 Training neural WAF model...\n";
    $epochLosses = $trainer->trainBatch($trainPairs, $maxEpochs, $targetLoss, 1);
    $trainedNetwork = $trainer->getTrainedNetwork();

    $trainTime = microtime(true) - $startTime;
    $epochsUsed = count($epochLosses);
    $finalLoss = end($epochLosses);

    // 4. Evaluate on training and unseen test datasets
    $trainEval = evaluateWafPerformance($trainedNetwork, $trainUnique);
    $testEval = evaluateWafPerformance($trainedNetwork, $testDataset);

    // 5. Benchmark inference speed
    $benchIterations = 500;
    $sampleFeatures = $testDataset[0]['input'];
    $benchStart = microtime(true);
    for ($i = 0; $i < $benchIterations; $i++) {
        $trainedNetwork->Calculate($sampleFeatures);
    }
    $benchElapsed = microtime(true) - $benchStart;
    $msPerReq = ($benchElapsed / $benchIterations) * 1000.0;
    $reqPerSec = $benchElapsed > 0 ? ($benchIterations / $benchElapsed) : 0.0;

    echo "\n✅ Training Complete!\n";
    echo "======================================================================\n";
    echo sprintf("   • Epochs used:          %d / %d\n", $epochsUsed, $maxEpochs);
    echo sprintf("   • Initial → Final MSE:  %.6f → %.6f\n", $epochLosses[0], $finalLoss);
    echo sprintf("   • Training Accuracy:    %d/%d (%.1f%%)\n", $trainEval['correct'], $trainEval['total'], $trainEval['accuracy'] * 100);
    echo sprintf("   • Unseen Test Accuracy: %d/%d (%.1f%%)\n", $testEval['correct'], $testEval['total'], $testEval['accuracy'] * 100);
    echo sprintf("   • Training Time:        %.2f sec\n", $trainTime);
    echo sprintf("   • Inference Latency:    %.3f ms/req (~%s req/sec)\n\n", $msPerReq, number_format((int)$reqPerSec));

    // 6. Print Per-Class Metrics on Unseen Test Set
    $classInfo = getWafClassInfo();
    echo "📈 Per-Class Metrics (Unseen Test Payloads):\n";
    echo "----------------------------------------------------------------------\n";
    echo " Class       | Precision | Recall | F1-Score | TP | FP | FN | Severity\n";
    echo "-------------+-----------+--------+----------+----+----+----+---------\n";
    foreach ($testEval['per_class'] as $c => $m) {
        $info = $classInfo[$c];
        echo sprintf(
            " [%d] %-7s |   %5.1f%%  | %5.1f%% |  %5.1f%%  | %2d | %2d | %2d | %s\n",
            $c,
            $info['code'],
            $m['precision'] * 100,
            $m['recall'] * 100,
            $m['f1'] * 100,
            $m['tp'],
            $m['fp'],
            $m['fn'],
            $info['severity']
        );
    }
    echo "----------------------------------------------------------------------\n\n";

    // 7. Print 5x5 Confusion Matrix (Unseen Test Set)
    echo "🧩 Confusion Matrix (Rows = Actual, Columns = Predicted):\n";
    echo "            ";
    for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
        echo sprintf(" %6s", $classInfo[$c]['code']);
    }
    echo "\n   ---------+" . str_repeat("-------", WAF_NUM_CLASSES) . "\n";
    for ($r = 0; $r < WAF_NUM_CLASSES; $r++) {
        echo sprintf("   %-8s |", $classInfo[$r]['code']);
        for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
            echo sprintf(" %6d", $testEval['confusion'][$r][$c]);
        }
        echo "\n";
    }
    echo "\n";

    // 8. Save trained model & sample batch file
    $modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'waf_model.json';
    $modelJson = $trainedNetwork->Json();
    file_put_contents($modelPath, $modelJson);

    $sampleBatch = array_map(fn($item) => [
        'payload' => $item['payload'],
        'expected_class' => $classInfo[$item['class']]['code'],
    ], $testDataset);
    $sampleBatchPath = __DIR__ . DIRECTORY_SEPARATOR . 'sample_payloads.json';
    file_put_contents($sampleBatchPath, json_encode($sampleBatch, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));

    $sizeKb = round(strlen($modelJson) / 1024, 1);
    echo "💾 Saved trained WAF model: {$modelPath} ({$sizeKb} KB)\n";
    echo "📁 Saved test payload suite: {$sampleBatchPath}\n\n";
    echo "Run WAF inspection with:\n";
    echo "  php examples/WAF/inspect_request.php\n";
    echo "  php examples/WAF/inspect_request.php \"1' UNION SELECT username,password FROM users--\"\n";
    echo "  php examples/WAF/inspect_request.php --file examples/WAF/sample_payloads.json\n\n";

} catch (\Throwable $e) {
    fwrite(STDERR, "❌ Error: " . $e->getMessage() . "\n");
    exit(1);
}
