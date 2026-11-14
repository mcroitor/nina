# NiNa Neural Network Library - Backpropagation & Optimization Guide

This guide covers the backpropagation training engine in NiNa, including loss functions, optimizers, learning rate schedules, weight initializers, and regularization.

---

## Table of Contents

- [Overview](#overview)
- [BackpropagationTrainer](#backpropagationtrainer)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
  - [SGD (with Momentum & Nesterov)](#sgd)
  - [Adam](#adam)
  - [RMSprop](#rmsprop)
- [Learning Rate Schedules](#learning-rate-schedules)
- [Regularization & Gradient Clipping](#regularization--gradient-clipping)
- [Weight Initializers](#weight-initializers)
- [Complete Training Example](#complete-training-example)

---

## Overview

The backpropagation algorithm in NiNa calculates analytical gradients of network loss with respect to all neuron weights via the chain rule. Weights are iteratively adjusted in the direction of steepest descent to minimize prediction error.

Key architectural highlights:
- **Heterogeneous Layers**: Full support for multi-layer networks with varying input and hidden dimensions.
- **Pluggable Optimizers**: Configurable algorithms (SGD, Adam, RMSprop) determining weight updates.
- **Dynamic Scheduling**: Learning rate adaptation across epochs.
- **Numerical Stability**: Gradient clipping by value and norm.

---

## BackpropagationTrainer

`BackpropagationTrainer` orchestrates forward caching, backward error propagation, and weight updates.

### Constructor

```php
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\schedule\StepDecaySchedule;

$trainer = new BackpropagationTrainer(
    $network,                     // Network instance
    new MeanSquaredError(),       // LossFunction
    0.01,                         // Initial learning rate
    new Adam(),                   // Optimizer (default: SGD)
    new StepDecaySchedule(10, 0.5) // Optional learning rate schedule
);
```

### Key Methods

- `trainSingle(array $input, array $expected): float`: Performs forward pass, loss calculation, backward pass, and immediate weight update for one example. Returns loss.
- `trainBatch(array $trainingData, int $epochs, ?float $targetLoss = null, int $batchSize = 1): array`: Trains across multiple epochs. When `batchSize > 1`, gradients are accumulated over mini-batches before updating weights.
- `computeGradients(array $input, array $expected): array`: Calculates and returns full network gradients without modifying weights.
- `getLossHistory(): array`: Returns loss recorded at each step.
- `setClipValue(?float $val)` / `setClipNorm(?float $norm)`: Configures gradient clipping limits.

---

## Loss Functions

NiNa provides standard loss functions under `mc\nina\loss\`:

| Class | Description | Standard Task |
|---|---|---|
| `MeanSquaredError` | $L = \frac{1}{n} \sum (y - \hat{y})^2$ | Regression, continuous fitting, XOR |
| `CrossEntropy` | $L = -\frac{1}{n} \sum y \log(\hat{y}) + (1-y)\log(1-\hat{y})$ | Binary & multi-class classification |

Both implement `LossFunction` providing `calculate(array $predicted, array $target): float` and `derivative(array $predicted, array $target): array`.

---

## Optimizers

Optimizers live under `mc\nina\optimizer\` and implement `Optimizer`.

### SGD

Stochastic Gradient Descent supporting classical momentum, Nesterov acceleration, and weight decay:

```php
use mc\nina\optimizer\SGD;

// SGD with 0.9 momentum and L2 weight decay of 0.001
$sgd = new SGD(
    momentum: 0.9,
    nesterov: true,
    weightDecay: 0.001,
    l1: 0.0
);
```

### Adam

Adaptive Moment Estimation maintaining exponential moving averages of gradients ($m$) and squared gradients ($v$):

```php
use mc\nina\optimizer\Adam;

$adam = new Adam(
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weightDecay: 0.0
);
```

### RMSprop

Root Mean Square Propagation dividing gradients by the running root-mean-square magnitude:

```php
use mc\nina\optimizer\RMSprop;

$rmsprop = new RMSprop(
    alpha: 0.99,
    epsilon: 1e-8,
    momentum: 0.0
);
```

---

## Learning Rate Schedules

Schedulers under `mc\nina\schedule\` adjust learning rate over time:

```php
use mc\nina\schedule\{ConstantSchedule, StepDecaySchedule, ExponentialSchedule, PlateauSchedule};

// Static learning rate
$schedule = new ConstantSchedule();

// Multiply learning rate by 0.5 every 10 epochs
$schedule = new StepDecaySchedule(stepSize: 10, gamma: 0.5);

// Multiply learning rate by 0.95 each epoch
$schedule = new ExponentialSchedule(gamma: 0.95);

// Halve learning rate when loss plateaus for 5 epochs
$schedule = new PlateauSchedule(factor: 0.5, patience: 5, minDelta: 1e-4);
```

---

## Regularization & Gradient Clipping

### L1 and L2 Regularization
Configured directly on the trainer or optimizer:

```php
$trainer->setWeightDecay(0.001);     // L2 weight decay (penalty on large weights)
$trainer->setL1Regularization(0.0001); // L1 Lasso (promotes weight sparsity)
```

### Gradient Clipping
Protects deep networks from numerical explosion:

```php
$trainer->setClipValue(1.0); // Clips each gradient element to [-1.0, 1.0]
$trainer->setClipNorm(5.0);  // Clips the total gradient vector norm to 5.0
```

---

## Weight Initializers

Proper weight initialization prevents vanishing or exploding activations. Available under `mc\nina\initializer\`:

```php
use mc\nina\initializer\{XavierInitializer, HeInitializer, RandomUniformInitializer};

// Xavier / Glorot initialization (recommended for Sigmoid and Tanh)
$network->initializeWeights(new XavierInitializer(uniform: true));

// He / Kaiming initialization (recommended for ReLU)
$network->initializeWeights(new HeInitializer(uniform: true));

// Bounded uniform random
$network->initializeWeights(new RandomUniformInitializer(min: -0.2, max: 0.2));
```

---

## Complete Training Example

```php
<?php
require_once __DIR__ . '/../src/autoload.php';

use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\activation\Sigmoid;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\schedule\StepDecaySchedule;
use mc\nina\initializer\XavierInitializer;

// 1. Build network: 2 inputs -> 4 hidden neurons -> 1 output
$network = Network::Create("Classifier", 2, 4, new Sigmoid());
$network->initializeWeights(new XavierInitializer());

// 2. Training data
$dataset = [
    [[0.1, 0.2], [0.1]],
    [[0.8, 0.9], [0.9]],
    [[0.2, 0.1], [0.1]],
    [[0.9, 0.8], [0.9]],
];

// 3. Configure trainer with Adam, decay schedule, and gradient clipping
$trainer = new BackpropagationTrainer(
    $network,
    new MeanSquaredError(),
    0.05,
    new Adam(),
    new StepDecaySchedule(stepSize: 20, gamma: 0.5)
);
$trainer->setClipValue(1.0);
$trainer->setWeightDecay(0.0001);

// 4. Train with mini-batch size of 2
$losses = $trainer->trainBatch($dataset, epochs: 100, targetLoss: 0.01, batchSize: 2);

echo "Final loss: " . end($losses) . "\n";
echo "Test prediction: " . $network->Calculate([0.85, 0.85])[0] . "\n";
```
