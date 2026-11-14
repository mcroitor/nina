<?php

/**
 * Unit tests for Neural Network Optimizers, Regularization, and Learning Rate Schedules
 * 
 * Tests SGD, Adam, RMSprop optimizers, L1/L2 regularization,
 * learning rate decay schedules, and their integration with BackpropagationTrainer.
 */

// include unit test framework
include_once __DIR__ . "/mc/unit/assert.php";
include_once __DIR__ . "/mc/unit/test.php";
include_once __DIR__ . "/mc/unit/framework.php";

use \mc\unit\Assert;
use \mc\unit\Test;
use \mc\unit\Framework;

// include autoloader
require_once __DIR__ . "/../src/autoload.php";

use mc\nina\Cell;
use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\activation\Sigmoid;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\SGD;
use mc\nina\optimizer\Adam;
use mc\nina\optimizer\RMSprop;
use mc\nina\schedule\ConstantSchedule;
use mc\nina\schedule\StepDecaySchedule;
use mc\nina\schedule\ExponentialSchedule;
use mc\nina\schedule\PlateauSchedule;
use mc\nina\exception\ValidationException;
use mc\nina\exception\ComputationException;

/**
 * Test Vanilla SGD and validation
 */
function test_sgd_basic_and_validation(): bool {
    $result = true;

    $sgd = new SGD();
    $result = $result && Assert::Equal('sgd', $sgd->getName());
    $result = $result && Assert::Equal(0.0, $sgd->getMomentum());
    $result = $result && Assert::Equal(false, $sgd->isNesterov());

    // Test parameter validation
    try {
        new SGD(-0.1);
        $result = false;
        Assert::True(false, "", "Should reject negative momentum");
    } catch (ValidationException $e) {
        Assert::True(true, "Correctly rejects negative momentum");
    }

    try {
        new SGD(1.0);
        $result = false;
        Assert::True(false, "", "Should reject momentum >= 1.0");
    } catch (ValidationException $e) {
        Assert::True(true, "Correctly rejects momentum >= 1.0");
    }

    // Test vanilla weight update: w_new = w_old - lr * g
    $cell = new Cell("sgd_cell_1", [1.0, 2.0]);
    $gradients = [0.5, -0.25];
    $lr = 0.1;

    $sgd->update($cell, $gradients, $lr);
    $dna = $cell->GetDna();

    $expected0 = 1.0 - 0.1 * 0.5;   // 0.95
    $expected1 = 2.0 - 0.1 * (-0.25); // 2.025

    $result = $result && Assert::True(abs($dna[0] - $expected0) < 1e-10, "Weight[0] matches SGD formula");
    $result = $result && Assert::True(abs($dna[1] - $expected1) < 1e-10, "Weight[1] matches SGD formula");

    return $result;
}

/**
 * Test SGD with Momentum and Nesterov
 */
function test_sgd_momentum_and_nesterov(): bool {
    $result = true;

    $cell = new Cell("momentum_cell", [0.5]);
    $sgd = new SGD(0.9, false);

    // Step 1: v_0 = 0.9 * 0 + 0.1 = 0.1, delta = -0.1 * 0.1 = -0.01, w = 0.49
    $sgd->update($cell, [0.1], 0.1);
    $dna1 = $cell->GetDna();
    $result = $result && Assert::True(abs($dna1[0] - 0.49) < 1e-10, "Momentum step 1 matches");

    // Step 2: v_1 = 0.9 * 0.1 + 0.1 = 0.19, delta = -0.1 * 0.19 = -0.019, w = 0.49 - 0.019 = 0.471
    $sgd->update($cell, [0.1], 0.1);
    $dna2 = $cell->GetDna();
    $result = $result && Assert::True(abs($dna2[0] - 0.471) < 1e-10, "Momentum step 2 accumulates");

    // Test reset clears velocity
    $sgd->reset();
    $cell->setDna([0.5]);
    $sgd->update($cell, [0.1], 0.1);
    $dnaAfterReset = $cell->GetDna();
    $result = $result && Assert::True(abs($dnaAfterReset[0] - 0.49) < 1e-10, "Reset cleared velocity");

    // Test Nesterov momentum
    $cellNesterov = new Cell("nesterov_cell", [0.5]);
    $sgdN = new SGD(0.9, true);
    $result = $result && Assert::True($sgdN->isNesterov(), "Nesterov flag is true");
    // Step 1: v = 0.9*0 + 0.1 = 0.1, step = 0.9*0.1 + 0.1 = 0.19, delta = -0.1 * 0.19 = -0.019
    $sgdN->update($cellNesterov, [0.1], 0.1);
    $dnaN = $cellNesterov->GetDna();
    $result = $result && Assert::True(abs($dnaN[0] - (0.5 - 0.019)) < 1e-10, "Nesterov step matches formula");

    return $result;
}

/**
 * Test L1 and L2 Regularization (Weight Decay)
 */
function test_regularization(): bool {
    $result = true;

    // L2 regularization: g_reg = g + lambda * w
    $cell = new Cell("reg_cell_l2", [2.0]);
    $sgd = new SGD(0.0, false, 0.01); // weightDecay = 0.01
    $result = $result && Assert::Equal(0.01, $sgd->getWeightDecay());

    // Gradient = 0.1. g_reg = 0.1 + 0.01 * 2.0 = 0.12. Update = -0.1 * 0.12 = -0.012. w = 1.988
    $sgd->update($cell, [0.1], 0.1);
    $dna = $cell->GetDna();
    $result = $result && Assert::True(abs($dna[0] - 1.988) < 1e-10, "L2 weight decay correctly penalizes large weights");

    // L1 regularization: g_reg = g + l1 * sign(w)
    $cellL1 = new Cell("reg_cell_l1", [2.0]);
    $sgdL1 = new SGD(0.0, false, 0.0, 0.05); // l1 = 0.05
    $result = $result && Assert::Equal(0.05, $sgdL1->getL1());

    // Gradient = 0.1. g_reg = 0.1 + 0.05 * 1.0 = 0.15. Update = -0.1 * 0.15 = -0.015. w = 1.985
    $sgdL1->update($cellL1, [0.1], 0.1);
    $dnaL1 = $cellL1->GetDna();
    $result = $result && Assert::True(abs($dnaL1[0] - 1.985) < 1e-10, "L1 regularization correctly applies sign penalty");

    return $result;
}

/**
 * Test Adam optimizer
 */
function test_adam_optimizer(): bool {
    $result = true;

    $adam = new Adam(0.9, 0.999, 1e-8);
    $result = $result && Assert::Equal('adam', $adam->getName());
    $result = $result && Assert::Equal(0.9, $adam->getBeta1());
    $result = $result && Assert::Equal(0.999, $adam->getBeta2());
    $result = $result && Assert::Equal(1e-8, $adam->getEpsilon());

    // Test parameter validation
    try {
        new Adam(0.0);
        $result = false;
        Assert::True(false, "", "Should reject beta1 <= 0");
    } catch (ValidationException $e) {
        Assert::True(true, "Correctly rejects invalid beta1");
    }

    try {
        new Adam(0.9, 1.0);
        $result = false;
        Assert::True(false, "", "Should reject beta2 >= 1");
    } catch (ValidationException $e) {
        Assert::True(true, "Correctly rejects invalid beta2");
    }

    // Step 1 update verification
    $cell = new Cell("adam_cell", [1.0]);
    $g = 0.2;
    $lr = 0.01;

    // m = (1-0.9)*0.2 = 0.02, m_hat = 0.02 / (1-0.9) = 0.2
    // v = (1-0.999)*(0.2^2) = 0.00004, v_hat = 0.00004 / (1-0.999) = 0.04
    // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.01 * 0.2 / (0.2 + 1e-8) ≈ 0.01
    // w_new ≈ 1.0 - 0.01 = 0.99
    $adam->update($cell, [$g], $lr);
    $dna = $cell->GetDna();
    $result = $result && Assert::True(abs($dna[0] - 0.99) < 1e-5, "Adam step 1 computes normalized update: {$dna[0]} ≈ 0.99");

    // Test Adam reset
    $adam->reset();
    $cell->setDna([1.0]);
    $adam->update($cell, [$g], $lr);
    $dnaReset = $cell->GetDna();
    $result = $result && Assert::True(abs($dnaReset[0] - 0.99) < 1e-5, "Adam reset re-initializes step count and moments");

    return $result;
}

/**
 * Test RMSprop optimizer
 */
function test_rmsprop_optimizer(): bool {
    $result = true;

    $rmsprop = new RMSprop(0.9, 1e-8, 0.0);
    $result = $result && Assert::Equal('rmsprop', $rmsprop->getName());
    $result = $result && Assert::Equal(0.9, $rmsprop->getAlpha());
    $result = $result && Assert::Equal(1e-8, $rmsprop->getEpsilon());

    // Test parameter validation
    try {
        new RMSprop(1.0);
        $result = false;
        Assert::True(false, "", "Should reject alpha >= 1");
    } catch (ValidationException $e) {
        Assert::True(true, "Correctly rejects invalid alpha");
    }

    // Step 1 update verification:
    // v = 0.9 * 0 + 0.1 * (0.4^2) = 0.016
    // normG = 0.4 / (sqrt(0.016) + 1e-8) = 0.4 / 0.1264911 = 3.1622776
    // delta = -0.01 * 3.1622776 = -0.0316228
    // w = 1.0 - 0.0316228 = 0.968377
    $cell = new Cell("rmsprop_cell", [1.0]);
    $rmsprop->update($cell, [0.4], 0.01);
    $dna = $cell->GetDna();
    $expected = 1.0 - 0.01 * (0.4 / sqrt(0.1 * 0.16));
    $result = $result && Assert::True(abs($dna[0] - $expected) < 1e-5, "RMSprop step matches expected value: {$dna[0]} ≈ {$expected}");

    return $result;
}

/**
 * Test Learning Rate Schedules
 */
function test_learning_rate_schedules(): bool {
    $result = true;

    // ConstantSchedule
    $const = new ConstantSchedule();
    $result = $result && Assert::Equal(0.05, $const->getLearningRate(0, 0.05));
    $result = $result && Assert::Equal(0.05, $const->getLearningRate(100, 0.05));

    // StepDecaySchedule: stepSize = 5, gamma = 0.5
    $step = new StepDecaySchedule(5, 0.5);
    $result = $result && Assert::Equal(0.1, $step->getLearningRate(0, 0.1));
    $result = $result && Assert::Equal(0.1, $step->getLearningRate(4, 0.1));
    $result = $result && Assert::Equal(0.05, $step->getLearningRate(5, 0.1));
    $result = $result && Assert::Equal(0.025, $step->getLearningRate(10, 0.1));

    // ExponentialSchedule: gamma = 0.9
    $exp = new ExponentialSchedule(0.9);
    $result = $result && Assert::Equal(0.1, $exp->getLearningRate(0, 0.1));
    $result = $result && Assert::True(abs($exp->getLearningRate(1, 0.1) - 0.09) < 1e-10, "Exponential decay at epoch 1");
    $result = $result && Assert::True(abs($exp->getLearningRate(2, 0.1) - 0.081) < 1e-10, "Exponential decay at epoch 2");

    // PlateauSchedule: patience = 2, factor = 0.5
    $plateau = new PlateauSchedule(0.5, 2, 0.01);
    $lr0 = $plateau->getLearningRate(0, 0.1, 1.0); // best = 1.0
    $result = $result && Assert::Equal(0.1, $lr0);

    $lr1 = $plateau->getLearningRate(1, 0.1, 0.995); // improved by < 0.01 (wait=1)
    $result = $result && Assert::Equal(0.1, $lr1);

    $lr2 = $plateau->getLearningRate(2, 0.1, 0.993); // wait=2 -> triggers decay!
    $result = $result && Assert::Equal(0.05, $lr2);

    return $result;
}

/**
 * Test BackpropagationTrainer integration with Adam, RMSprop, Schedulers, and Mini-batching
 */
function test_backprop_trainer_full_integration(): bool {
    $result = true;

    $network = Network::Create("integration_net", 2, 2, new Sigmoid());
    $adam = new Adam(0.9, 0.999);
    $stepSchedule = new StepDecaySchedule(2, 0.5);

    $trainer = new BackpropagationTrainer(
        $network,
        new MeanSquaredError(),
        0.1,
        $adam,
        $stepSchedule
    );

    $result = $result && Assert::Equal('adam', $trainer->getOptimizer()->getName());
    $result = $result && Assert::True($trainer->getLearningRateSchedule() instanceof StepDecaySchedule, "Schedule is StepDecay");

    // Test computeGradients method without updating weights
    $initialWeights = $network->GetLayer(0)->GetCell(0)->GetDna();
    $grads = $trainer->computeGradients([0.5, 0.5], [1.0, 0.0]);
    $weightsAfterGrad = $network->GetLayer(0)->GetCell(0)->GetDna();

    $result = $result && Assert::Equal(count($grads), 2); // 2 layers
    $result = $result && Assert::Equal($initialWeights, $weightsAfterGrad); // Weights unchanged

    // Test mini-batch training (batchSize = 2)
    $trainingData = [
        [[0.1, 0.2], [0.8, 0.3]],
        [[0.4, 0.5], [0.2, 0.9]],
        [[0.7, 0.1], [0.9, 0.1]],
        [[0.3, 0.8], [0.1, 0.6]],
    ];

    $losses = $trainer->trainBatch($trainingData, 4, null, 2); // 4 epochs, batchSize = 2
    $result = $result && Assert::Equal(4, count($losses));
    $lossDecreasedAdam = end($losses) < $losses[0];
    $result = $result && Assert::True($lossDecreasedAdam, "Loss decreased with Adam and mini-batching", "Adam loss did not decrease: " . end($losses) . " >= " . $losses[0]);

    // Test learning rate decay during training
    // At epoch 0, 1: lr = 0.1. At epoch 2, 3: lr = 0.05.
    $currentLr = $trainer->getLearningRate();
    $result = $result && Assert::True($currentLr <= 0.05, "Learning rate scheduled down correctly", "LR not <= 0.05: {$currentLr}");

    // Test RMSprop integration
    $networkRMS = Network::Create("rmsprop_net", 2, 2, new Sigmoid());
    $trainerRMS = new BackpropagationTrainer($networkRMS, new MeanSquaredError(), 0.01);
    $trainerRMS->setOptimizer(new RMSprop(0.95));
    $trainerRMS->setWeightDecay(0.001);
    $result = $result && Assert::Equal('rmsprop', $trainerRMS->getOptimizer()->getName());
    $result = $result && Assert::Equal(0.001, $trainerRMS->getWeightDecay());

    $lossesRMS = $trainerRMS->trainBatch($trainingData, 10);
    $lossDecreasedRMS = end($lossesRMS) < $lossesRMS[0];
    $result = $result && Assert::True($lossDecreasedRMS, "Loss decreased with RMSprop", "RMSprop loss did not decrease: " . end($lossesRMS) . " >= " . $lossesRMS[0]);

    return $result;
}

$framework = new Framework();

$framework->AddTest("test_sgd_basic_and_validation");
$framework->AddTest("test_sgd_momentum_and_nesterov");
$framework->AddTest("test_regularization");
$framework->AddTest("test_adam_optimizer");
$framework->AddTest("test_rmsprop_optimizer");
$framework->AddTest("test_learning_rate_schedules");
$framework->AddTest("test_backprop_trainer_full_integration");

echo "=== Testing Neural Network Optimizers, Regularization, and Schedules ===\n\n";

$framework->Run();
$framework->PrintInfo();

if ($framework->TotalPassed() === $framework->TotalTests()) {
    echo "\n🎉 All Optimizer and Schedule tests passed!\n";
    echo "✅ SGD with momentum and Nesterov works correctly\n";
    echo "✅ L1 and L2 weight decay regularizations function properly\n";
    echo "✅ Adam optimizer computes adaptive moments correctly\n";
    echo "✅ RMSprop optimizer adapts learning rates accurately\n";
    echo "✅ Constant, StepDecay, Exponential, and Plateau schedules work\n";
    echo "✅ Mini-batch gradient accumulation and full integration verified\n\n";
} else {
    echo "\n❌ Some optimizer tests failed.\n";
}
