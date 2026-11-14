<?php

/**
 * Multi-Class Digit Recognition (0-9) - Inference & CLI Showcase
 *
 * Classifies 5x7 digit patterns using the trained heterogeneous neural network.
 *
 * Usage:
 *   1. Full 0-9 visual showcase (clean + noisy):
 *      php examples/digit_recognition/recognize_digit.php
 *
 *   2. Recognize from a pattern file (.txt or .json):
 *      php examples/digit_recognition/recognize_digit.php examples/digit_recognition/patterns/digit_5.txt
 *
 *   3. Recognize a specific digit with optional synthetic noise:
 *      php examples/digit_recognition/recognize_digit.php --digit 7 --noise 0.2
 */

require_once __DIR__ . '/common.php';

use mc\nina\Network;

$modelFile = __DIR__ . DIRECTORY_SEPARATOR . 'digit_model.json';
if (!file_exists($modelFile)) {
    fwrite(STDERR, "Model file not found: {$modelFile}\n");
    fwrite(STDERR, "Please run 'php examples/digit_recognition/train_digits.php' first.\n");
    exit(1);
}

$network = Network::Load((string)file_get_contents($modelFile));

// Parse CLI arguments
if ($argc > 1) {
    $arg1 = $argv[1];

    if ($arg1 === '--help' || $arg1 === '-h') {
        printUsage();
        exit(0);
    }

    if ($arg1 === '--digit' || $arg1 === '-d') {
        $digit = isset($argv[2]) ? (int)$argv[2] : 0;
        if ($digit < 0 || $digit > 9) {
            fwrite(STDERR, "Error: --digit must be between 0 and 9.\n");
            exit(1);
        }
        $noise = 0.0;
        for ($i = 3; $i < $argc; $i++) {
            if (($argv[$i] === '--noise' || $argv[$i] === '-n') && isset($argv[$i + 1])) {
                $noise = max(0.0, min(0.5, (float)$argv[$i + 1]));
            }
        }

        $basePatterns = getBaseDigitPatterns();
        $pattern = addDigitNoise($basePatterns[$digit], $noise);
        $label = "Digit {$digit}" . ($noise > 0 ? " (noise={$noise})" : " (clean)");
        inspectSinglePattern($network, $pattern, $label, $digit);
        exit(0);
    }

    // Load from file path
    $pattern = loadDigitPatternFile($arg1);
    inspectSinglePattern($network, $pattern, "File: " . basename($arg1));
    exit(0);
}

// Default: run showcase across digits 0-9
runShowcase($network);

/**
 * Displays detailed multi-class classification report for a single 5x7 pattern.
 */
function inspectSinglePattern(Network $network, array $pattern, string $title, ?int $expectedDigit = null): void
{
    echo "\n============================================================\n";
    echo "   🔎 Digit Recognition Analysis — {$title}\n";
    echo "============================================================\n\n";

    echo "Input Sprite (5×7):\n";
    echo renderDigitAscii($pattern, '    ');
    echo "\n";

    $result = predictDigit($network, $pattern);
    $predicted = $result['predicted_digit'];
    $confidence = $result['confidence'] * 100;

    echo "Class Activation Distribution (0–9):\n";
    echo "------------------------------------------------------------\n";
    for ($d = 0; $d < DIGIT_CLASSES; $d++) {
        $score = $result['raw_scores'][$d];
        $barLen = (int)round(max(0.0, min(1.0, $score)) * 25);
        $bar = str_repeat('█', $barLen) . str_repeat('░', 25 - $barLen);
        $marker = ($d === $predicted) ? ' ◄ PREDICTED' : '';
        echo sprintf("  [%d] |%s| %5.1f%%%s\n", $d, $bar, $score * 100, $marker);
    }
    echo "------------------------------------------------------------\n";

    echo sprintf("\n🎯 Predicted Digit: %d (Confidence: %.1f%%)\n", $predicted, $confidence);
    if ($expectedDigit !== null) {
        $status = ($predicted === $expectedDigit) ? '✅ CORRECT' : "❌ MISMATCH (expected {$expectedDigit})";
        echo "   Verification:    {$status}\n";
    }
    echo "\n";
}

/**
 * Runs a visual showcase on all digits 0-9 (clean + noisy samples).
 */
function runShowcase(Network $network): void
{
    echo "\n============================================================\n";
    echo "   🔢 NiNa Multi-Class Digit Recognition (0–9) Showcase\n";
    echo "============================================================\n";
    echo "Loaded Network: {$network->GetName()} ({$network->GetNrLayers()} layers)\n\n";

    $basePatterns = getBaseDigitPatterns();
    $variantPatterns = getVariantDigitPatterns();

    $correctClean = 0;
    $correctVariant = 0;
    $correctNoisy = 0;

    echo "1️⃣  Testing Canonical 5×7 Digits (0–9):\n";
    echo "------------------------------------------------------------\n";
    foreach ($basePatterns as $digit => $pattern) {
        $res = predictDigit($network, $pattern);
        $pred = $res['predicted_digit'];
        if ($pred === $digit) {
            $correctClean++;
        }
        $status = ($pred === $digit) ? '✅' : '❌';
        echo sprintf(
            "  %s Digit %d → Predicted: %d (confidence: %5.1f%%)\n",
            $status,
            $digit,
            $pred,
            $res['confidence'] * 100
        );
    }

    echo "\n2️⃣  Testing Alternative LCD Font Variants (0–9):\n";
    echo "------------------------------------------------------------\n";
    foreach ($variantPatterns as $digit => $pattern) {
        $res = predictDigit($network, $pattern);
        $pred = $res['predicted_digit'];
        if ($pred === $digit) {
            $correctVariant++;
        }
        $status = ($pred === $digit) ? '✅' : '❌';
        echo sprintf(
            "  %s Variant %d → Predicted: %d (confidence: %5.1f%%)\n",
            $status,
            $digit,
            $pred,
            $res['confidence'] * 100
        );
    }

    echo "\n3️⃣  Testing Noisy Digits (15% Pixel Corruption):\n";
    echo "------------------------------------------------------------\n";
    foreach ($basePatterns as $digit => $pattern) {
        $noisy = addDigitNoise($pattern, 0.15);
        $res = predictDigit($network, $noisy);
        $pred = $res['predicted_digit'];
        if ($pred === $digit) {
            $correctNoisy++;
        }
        $status = ($pred === $digit) ? '✅' : '❌';
        echo sprintf(
            "  %s Noisy %d   → Predicted: %d (confidence: %5.1f%%)\n",
            $status,
            $digit,
            $pred,
            $res['confidence'] * 100
        );
    }

    // Show one detailed visual example
    $sampleNoisy8 = addDigitNoise($basePatterns[8], 0.15);
    inspectSinglePattern($network, $sampleNoisy8, "Sample Noisy Digit 8 (15% noise)", 8);

    echo "📊 Summary:\n";
    echo sprintf("   • Canonical Digits Accuracy: %d/10 (%d%%)\n", $correctClean, $correctClean * 10);
    echo sprintf("   • LCD Font Variant Accuracy: %d/10 (%d%%)\n", $correctVariant, $correctVariant * 10);
    echo sprintf("   • Noisy Digits Accuracy:     %d/10 (%d%%)\n\n", $correctNoisy, $correctNoisy * 10);
}

function printUsage(): void
{
    echo <<<USAGE

NiNa Multi-Class Digit Recognition (0-9)
========================================

Usage:
  php examples/digit_recognition/recognize_digit.php
  php examples/digit_recognition/recognize_digit.php <pattern_file>
  php examples/digit_recognition/recognize_digit.php --digit <0-9> [--noise <0.0-0.5>]

Examples:
  php examples/digit_recognition/recognize_digit.php
  php examples/digit_recognition/recognize_digit.php examples/digit_recognition/patterns/digit_4.txt
  php examples/digit_recognition/recognize_digit.php --digit 9 --noise 0.2

USAGE;
}
