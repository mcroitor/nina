<?php

/**
 * Multi-Class Digit Recognition (0-9) - Common Functions & Dataset
 *
 * Provides 5x7 bitmap representations for digits 0-9, network construction
 * with heterogeneous layers (35 inputs -> 24 hidden Tanh -> 10 Sigmoid outputs),
 * noise generation, ASCII rendering, and evaluation utilities.
 *
 * @package NiNa\Examples\DigitRecognition
 */

require_once __DIR__ . '/../../src/autoload.php';

use mc\nina\Cell;
use mc\nina\Layer;
use mc\nina\Network;
use mc\nina\activation\Tanh;
use mc\nina\activation\Sigmoid;
use mc\nina\initializer\XavierInitializer;

const DIGIT_WIDTH = 5;
const DIGIT_HEIGHT = 7;
const DIGIT_INPUT_SIZE = DIGIT_WIDTH * DIGIT_HEIGHT; // 35 pixels
const DIGIT_CLASSES = 10;

/**
 * Returns canonical 5x7 pixel bitmaps (35 elements each) for digits 0-9.
 *
 * @return array<int, array<float>>
 */
function getBaseDigitPatterns(): array
{
    return [
        0 => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,1,1,
            1,0,1,0,1,
            1,1,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        1 => [
            0,0,1,0,0,
            0,1,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,1,1,1,0,
        ],
        2 => [
            0,1,1,1,0,
            1,0,0,0,1,
            0,0,0,0,1,
            0,0,1,1,0,
            0,1,0,0,0,
            1,0,0,0,0,
            1,1,1,1,1,
        ],
        3 => [
            0,1,1,1,0,
            1,0,0,0,1,
            0,0,0,0,1,
            0,0,1,1,0,
            0,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        4 => [
            0,0,0,1,0,
            0,0,1,1,0,
            0,1,0,1,0,
            1,0,0,1,0,
            1,1,1,1,1,
            0,0,0,1,0,
            0,0,0,1,0,
        ],
        5 => [
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,0,
            0,0,0,0,1,
            0,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        6 => [
            0,0,1,1,0,
            0,1,0,0,0,
            1,0,0,0,0,
            1,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        7 => [
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
            0,1,0,0,0,
        ],
        8 => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        9 => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            0,1,1,0,0,
        ],
    ];
}

/**
 * Returns secondary style variations (5x7) for digits 0-9 (e.g. LCD/7-segment style).
 *
 * @return array<int, array<float>>
 */
function getVariantDigitPatterns(): array
{
    return [
        0 => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        1 => [
            0,0,1,0,0,
            0,1,1,0,0,
            1,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            1,1,1,1,1,
        ],
        2 => [
            1,1,1,1,0,
            0,0,0,0,1,
            0,0,0,0,1,
            0,1,1,1,0,
            1,0,0,0,0,
            1,0,0,0,0,
            1,1,1,1,1,
        ],
        3 => [
            1,1,1,1,0,
            0,0,0,0,1,
            0,0,0,0,1,
            0,1,1,1,0,
            0,0,0,0,1,
            0,0,0,0,1,
            1,1,1,1,0,
        ],
        4 => [
            1,0,0,1,0,
            1,0,0,1,0,
            1,0,0,1,0,
            1,1,1,1,1,
            0,0,0,1,0,
            0,0,0,1,0,
            0,0,0,1,0,
        ],
        5 => [
            1,1,1,1,1,
            1,0,0,0,0,
            1,1,1,1,0,
            0,0,0,0,1,
            0,0,0,0,1,
            0,0,0,0,1,
            1,1,1,1,0,
        ],
        6 => [
            0,1,1,1,0,
            1,0,0,0,0,
            1,0,0,0,0,
            1,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
        7 => [
            1,1,1,1,1,
            0,0,0,0,1,
            0,0,0,1,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
            0,0,1,0,0,
        ],
        8 => [
            1,1,1,1,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1,
            1,0,0,0,1,
            1,0,0,0,1,
            1,1,1,1,1,
        ],
        9 => [
            0,1,1,1,0,
            1,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,1,
            0,0,0,0,1,
            1,0,0,0,1,
            0,1,1,1,0,
        ],
    ];
}

/**
 * Creates a one-hot target vector of length 10 for the given digit (0-9).
 *
 * @param int $digit Target digit (0-9)
 * @return array<float>
 */
function createOneHotTarget(int $digit): array
{
    $target = array_fill(0, DIGIT_CLASSES, 0.0);
    $target[$digit] = 1.0;
    return $target;
}

/**
 * Adds bounded uniform noise to a pixel pattern.
 *
 * @param array<float> $pattern Input pattern with values in [0, 1]
 * @param float $noiseLevel Maximum noise amplitude (0.0 to 1.0)
 * @return array<float>
 */
function addDigitNoise(array $pattern, float $noiseLevel = 0.1): array
{
    if ($noiseLevel <= 0.0) {
        return array_map('floatval', $pattern);
    }

    $noisy = [];
    foreach ($pattern as $pixel) {
        $delta = (mt_rand() / mt_getrandmax() - 0.5) * 2.0 * $noiseLevel;
        $noisy[] = max(0.0, min(1.0, (float)$pixel + $delta));
    }
    return $noisy;
}

/**
 * Builds a multi-class digit recognition network with heterogeneous layers:
 * - Hidden Layer: $hiddenSize neurons (35 inputs each, Tanh activation)
 * - Output Layer: 10 neurons ($hiddenSize inputs each, Sigmoid activation)
 * Initialized with XavierInitializer.
 *
 * @param int $hiddenSize Number of neurons in the hidden layer
 * @return Network
 */
function createDigitNetwork(int $hiddenSize = 24): Network
{
    $hiddenCells = [];
    for ($i = 0; $i < $hiddenSize; $i++) {
        $hiddenCells[] = Cell::Create("hidden_{$i}", DIGIT_INPUT_SIZE, 0.0, new Tanh());
    }
    $hiddenLayer = new Layer("hidden_layer", $hiddenCells);

    $outputCells = [];
    for ($d = 0; $d < DIGIT_CLASSES; $d++) {
        $outputCells[] = Cell::Create("digit_{$d}", $hiddenSize, 0.0, new Sigmoid());
    }
    $outputLayer = new Layer("output_layer", $outputCells);

    $network = new Network("digit_classifier_5x7", [$hiddenLayer, $outputLayer]);
    $network->initializeWeights(new XavierInitializer());

    return $network;
}

/**
 * Generates a balanced multi-class training dataset for digits 0-9.
 *
 * @param int $samplesPerDigit Number of training samples per digit (minimum 2)
 * @param float $noiseLevel Noise level applied to augmented samples
 * @return array<int, array{input: array<float>, target: array<float>, digit: int}>
 */
function generateDigitDataset(int $samplesPerDigit = 8, float $noiseLevel = 0.08): array
{
    $basePatterns = getBaseDigitPatterns();
    $variantPatterns = getVariantDigitPatterns();
    $dataset = [];

    for ($s = 0; $s < $samplesPerDigit; $s++) {
        for ($digit = 0; $digit < DIGIT_CLASSES; $digit++) {
            $source = ($s % 2 === 0) ? $basePatterns[$digit] : $variantPatterns[$digit];
            // Keep first 2 passes clean (canonical + variant), add noise to remaining passes
            $input = ($s < 2) ? array_map('floatval', $source) : addDigitNoise($source, $noiseLevel);

            $dataset[] = [
                'input' => $input,
                'target' => createOneHotTarget($digit),
                'digit' => $digit,
            ];
        }
    }

    return $dataset;
}

/**
 * Formats dataset records into [input, target] tuples for BackpropagationTrainer::trainBatch().
 *
 * @param array $dataset Dataset from generateDigitDataset()
 * @return array<int, array{0: array<float>, 1: array<float>}>
 */
function toBackpropPairs(array $dataset): array
{
    $pairs = [];
    foreach ($dataset as $item) {
        $pairs[] = [$item['input'], $item['target']];
    }
    return $pairs;
}

/**
 * Predicts digit class (0-9) and returns sorted rankings and normalized probabilities.
 *
 * @param Network $network Trained digit network
 * @param array<float> $input 35-element pixel array
 * @return array{predicted_digit: int, confidence: float, raw_scores: array<float>, rankings: array<int, float>}
 */
function predictDigit(Network $network, array $input): array
{
    $scores = $network->Calculate($input);

    $rankings = [];
    foreach ($scores as $digit => $score) {
        $rankings[$digit] = (float)$score;
    }
    arsort($rankings);

    $predictedDigit = (int)array_key_first($rankings);
    $bestScore = $rankings[$predictedDigit];

    return [
        'predicted_digit' => $predictedDigit,
        'confidence' => $bestScore,
        'raw_scores' => $scores,
        'rankings' => $rankings,
    ];
}

/**
 * Evaluates multi-class top-1 accuracy on a dataset.
 *
 * @param Network $network
 * @param array $dataset
 * @return array{accuracy: float, correct: int, total: int, confusion: array<int, array<int, int>>}
 */
function evaluateDigitDataset(Network $network, array $dataset): array
{
    $correct = 0;
    $total = count($dataset);
    $confusion = array_fill(0, DIGIT_CLASSES, array_fill(0, DIGIT_CLASSES, 0));

    foreach ($dataset as $sample) {
        $res = predictDigit($network, $sample['input']);
        $pred = $res['predicted_digit'];
        $expected = $sample['digit'];
        $confusion[$expected][$pred]++;
        if ($pred === $expected) {
            $correct++;
        }
    }

    return [
        'accuracy' => $total > 0 ? ($correct / $total) : 0.0,
        'correct' => $correct,
        'total' => $total,
        'confusion' => $confusion,
    ];
}

/**
 * Renders a 5x7 digit pattern as ASCII art.
 *
 * @param array<float> $pattern 35-element array
 * @param string $indent Optional line prefix
 * @return string
 */
function renderDigitAscii(array $pattern, string $indent = '  '): string
{
    $out = '';
    for ($row = 0; $row < DIGIT_HEIGHT; $row++) {
        $out .= $indent;
        for ($col = 0; $col < DIGIT_WIDTH; $col++) {
            $val = $pattern[$row * DIGIT_WIDTH + $col] ?? 0.0;
            if ($val >= 0.65) {
                $out .= '██';
            } elseif ($val >= 0.25) {
                $out .= '▒▒';
            } else {
                $out .= '··';
            }
        }
        $out .= "\n";
    }
    return $out;
}

/**
 * Loads a 5x7 digit pattern from a text file (0/1 or ./#) or JSON array.
 *
 * @param string $filePath
 * @return array<float>
 */
function loadDigitPatternFile(string $filePath): array
{
    if (!file_exists($filePath)) {
        throw new RuntimeException("Pattern file not found: {$filePath}");
    }

    $content = trim((string)file_get_contents($filePath));
    $decoded = json_decode($content, true);
    if (is_array($decoded)) {
        $values = array_map('floatval', $decoded);
        return array_slice(array_pad($values, DIGIT_INPUT_SIZE, 0.0), 0, DIGIT_INPUT_SIZE);
    }

    $pixels = [];
    $len = strlen($content);
    for ($i = 0; $i < $len; $i++) {
        $ch = $content[$i];
        if ($ch === '1' || $ch === '#' || $ch === '@') {
            $pixels[] = 1.0;
        } elseif ($ch === '0' || $ch === '.') {
            $pixels[] = 0.0;
        }
    }

    if (count($pixels) !== DIGIT_INPUT_SIZE) {
        throw new RuntimeException(
            "Expected " . DIGIT_INPUT_SIZE . " pixels (5x7), found " . count($pixels) . " in {$filePath}"
        );
    }

    return $pixels;
}
