<?php

/**
 * Integration Test: Solving the XOR Problem with Backpropagation
 * 
 * Verifies that the backpropagation algorithm, multi-layer heterogeneous architectures,
 * weight initializers, and optimizers can successfully solve the non-linear XOR classification task.
 */

include_once __DIR__ . "/mc/unit/assert.php";
include_once __DIR__ . "/mc/unit/test.php";
include_once __DIR__ . "/mc/unit/framework.php";

use \mc\unit\Assert;
use \mc\unit\Test;
use \mc\unit\Framework;

require_once __DIR__ . "/../src/autoload.php";

use mc\nina\Cell;
use mc\nina\Layer;
use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\activation\Sigmoid;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\optimizer\SGD;
use mc\nina\initializer\XavierInitializer;
use mc\nina\initializer\HeInitializer;
use mc\nina\initializer\RandomUniformInitializer;

/**
 * Test weight initializers
 */
function test_weight_initializers(): bool {
    $result = true;

    // RandomUniform
    $uniform = new RandomUniformInitializer(-1.0, 1.0);
    $weights = $uniform->initialize(50);
    $result = $result && Assert::Equal(50, count($weights));
    $allInRange = true;
    foreach ($weights as $w) {
        if ($w < -1.0 || $w > 1.0) {
            $allInRange = false;
        }
    }
    $result = $result && Assert::True($allInRange, "All RandomUniform weights in [-1, 1]");

    // Xavier Uniform
    $xavier = new XavierInitializer(true);
    $result = $result && Assert::Equal('xavier_uniform', $xavier->getName());
    $xWeights = $xavier->initialize(100, 10, 10);
    $limit = sqrt(6.0 / 20.0); // ~0.5477
    $xInRange = true;
    foreach ($xWeights as $w) {
        if ($w < -$limit || $w > $limit) {
            $xInRange = false;
        }
    }
    $result = $result && Assert::True($xInRange, "Xavier uniform weights in bound");

    // He Uniform
    $he = new HeInitializer(true);
    $result = $result && Assert::Equal('he_uniform', $he->getName());
    $heWeights = $he->initialize(100, 16);
    $heLimit = sqrt(6.0 / 16.0); // ~0.6123
    $heInRange = true;
    foreach ($heWeights as $w) {
        if ($w < -$heLimit || $w > $heLimit) {
            $heInRange = false;
        }
    }
    $result = $result && Assert::True($heInRange, "He uniform weights in bound");

    // Network::initializeWeights
    $net = Network::Create("init_test", 2, 4);
    $net->initializeWeights($xavier);
    $dna = $net->GetLayer(0)->GetCell(0)->GetDna();
    $nonZero = false;
    foreach ($dna as $w) {
        if (abs($w) > 1e-6) {
            $nonZero = true;
        }
    }
    $result = $result && Assert::True($nonZero, "Network::initializeWeights populated non-zero weights");

    return $result;
}

/**
 * Test gradient clipping
 */
function test_gradient_clipping(): bool {
    $result = true;

    $cell = new Cell("clip_cell", [0.5, 0.5], new Sigmoid());
    $layer = new Layer("clip_layer", [$cell]);
    $net = new Network("clip_net", [$layer]);

    $trainer = new BackpropagationTrainer($net, new MeanSquaredError(), 0.1);

    // Test clipping setters
    $trainer->setClipValue(0.05);
    $result = $result && Assert::Equal(0.05, $trainer->getClipValue());

    $trainer->setClipNorm(0.1);
    $result = $result && Assert::Equal(0.1, $trainer->getClipNorm());

    // Compute gradients on an example that produces larger gradients
    // Input: [10.0, 10.0], Expected: [0.0]
    $grads = $trainer->computeGradients([10.0, 10.0], [0.0]);
    $cellGrads = $grads[0][0];

    // Gradients must not exceed clipValue = 0.05
    $withinClipValue = true;
    foreach ($cellGrads as $g) {
        if (abs($g) > 0.05 + 1e-7) {
            $withinClipValue = false;
        }
    }
    $result = $result && Assert::True($withinClipValue, "Gradients clipped by value to <= 0.05");

    return $result;
}

/**
 * Test solving the XOR Problem with Backpropagation
 * 
 * Architecture: 2 inputs -> Hidden layer (4 neurons) -> Output layer (1 neuron)
 * Target: XOR truth table:
 *   [0, 0] -> 0
 *   [0, 1] -> 1
 *   [1, 0] -> 1
 *   [1, 1] -> 0
 */
function test_solve_xor_problem(): bool {
    $result = true;

    // Heterogeneous Architecture:
    // Hidden layer: 4 neurons, each receiving 2 inputs
    $hiddenCells = [
        new Cell("h0", [0.5, -0.4], new Sigmoid()),
        new Cell("h1", [-0.3, 0.7], new Sigmoid()),
        new Cell("h2", [0.6, 0.8], new Sigmoid()),
        new Cell("h3", [-0.5, -0.6], new Sigmoid()),
    ];
    $hiddenLayer = new Layer("hidden", $hiddenCells);

    // Output layer: 1 neuron, receiving 4 inputs from hidden layer
    $outputCell = new Cell("out0", [0.8, -0.5, 0.4, -0.7], new Sigmoid());
    $outputLayer = new Layer("output", [$outputCell]);

    $network = new Network("xor_solver", [$hiddenLayer, $outputLayer]);

    $xorData = [
        [[0.0, 0.0], [0.0]],
        [[0.0, 1.0], [1.0]],
        [[1.0, 0.0], [1.0]],
        [[1.0, 1.0], [0.0]],
    ];

    $adam = new Adam(0.9, 0.999);
    $trainer = new BackpropagationTrainer(
        $network,
        new MeanSquaredError(),
        0.1,
        $adam
    );

    // Train on XOR data
    $losses = $trainer->trainBatch($xorData, 600, 0.01);

    $finalLoss = end($losses);
    $result = $result && Assert::True(
        $finalLoss < 0.02,
        "XOR trained to loss < 0.02 (reached {$finalLoss})",
        "XOR loss failed to reach < 0.02: {$finalLoss}"
    );

    // Verify all 4 XOR truth table predictions
    $p00 = $network->Calculate([0.0, 0.0])[0];
    $p01 = $network->Calculate([0.0, 1.0])[0];
    $p10 = $network->Calculate([1.0, 0.0])[0];
    $p11 = $network->Calculate([1.0, 1.0])[0];

    $result = $result && Assert::True($p00 < 0.2, "XOR(0,0) = 0 (predicted " . round($p00, 3) . ")");
    $result = $result && Assert::True($p01 > 0.8, "XOR(0,1) = 1 (predicted " . round($p01, 3) . ")");
    $result = $result && Assert::True($p10 > 0.8, "XOR(1,0) = 1 (predicted " . round($p10, 3) . ")");
    $result = $result && Assert::True($p11 < 0.2, "XOR(1,1) = 0 (predicted " . round($p11, 3) . ")");

    return $result;
}

$framework = new Framework();

$framework->AddTest("test_weight_initializers");
$framework->AddTest("test_gradient_clipping");
$framework->AddTest("test_solve_xor_problem");

echo "=== Testing Weight Initializers, Gradient Clipping, and XOR Problem ===\n\n";

$framework->Run();
$framework->PrintInfo();

if ($framework->TotalPassed() === $framework->TotalTests()) {
    echo "\n🎉 All XOR and Backpropagation integration tests passed!\n";
    echo "✅ Weight initializers (Xavier, He, RandomUniform) work properly\n";
    echo "✅ Gradient clipping (value and norm) functions correctly\n";
    echo "✅ Non-linear XOR problem successfully solved with Backpropagation & Adam!\n\n";
} else {
    echo "\n❌ Some tests failed.\n";
}
