<?php
/**
 * NiNa Neural Network Library - Autoloader
 * 
 * This autoloader provides automatic class loading for the NiNa neural network library.
 * It handles all core classes, activation functions, loss functions, and utilities.
 * 
 * Usage:
 *   require_once __DIR__ . '/autoload.php';
 * 
 * After including this file, all NiNa classes will be automatically loaded when needed.
 * 
 * @package NiNa
 * @author Mihail Croitor
 * @version 1.0.0
 * @since 1.0.0
 */

// Define base directory for the library
define('NINA_BASE_DIR', __DIR__);

/**
 * Custom autoloader for NiNa neural network library
 * 
 * @param string $className The fully qualified class name to load
 * @return void
 */
spl_autoload_register(function ($className) {
    // Only handle classes from the mc\nina namespace
    if (strpos($className, 'mc\\nina\\') !== 0) {
        return;
    }
    
    // Convert namespace to file path
    $relativePath = str_replace('\\', DIRECTORY_SEPARATOR, $className);
    $relativePath = str_replace('mc' . DIRECTORY_SEPARATOR, 'mc' . DIRECTORY_SEPARATOR, $relativePath);
    
    // Build full file path
    $filePath = NINA_BASE_DIR . DIRECTORY_SEPARATOR . $relativePath . '.php';
    
    // Load file if it exists
    if (file_exists($filePath)) {
        require_once $filePath;
    }
});

// Load utility functions
require_once NINA_BASE_DIR . '/mc/lib.php';

// Pre-load core interfaces and base classes to ensure proper inheritance
$coreFiles = [
    // Activation function interface
    '/mc/nina/activation/ActivationFunction.php',
    
    // Exception classes
    '/mc/nina/exception/NinaException.php',
    '/mc/nina/exception/ValidationException.php',
    '/mc/nina/exception/ComputationException.php',
    '/mc/nina/exception/SerializationException.php',
    '/mc/nina/exception/NetworkException.php',
    '/mc/nina/exception/LayerException.php',

    // Core classes
    '/mc/nina/cell.php',
    '/mc/nina/layer.php',
    '/mc/nina/network.php',

    // Activation functions
    '/mc/nina/activation/Sigmoid.php',
    '/mc/nina/activation/ReLU.php',
    '/mc/nina/activation/Tanh.php',
    '/mc/nina/activation/Softmax.php',

    // Loss functions
    '/mc/nina/loss/LossFunction.php',
    '/mc/nina/loss/MeanSquaredError.php',
    '/mc/nina/loss/CrossEntropy.php',

    // Optimizers
    '/mc/nina/optimizer/Optimizer.php',
    '/mc/nina/optimizer/AbstractOptimizer.php',
    '/mc/nina/optimizer/SGD.php',
    '/mc/nina/optimizer/Adam.php',
    '/mc/nina/optimizer/RMSprop.php',

    // Learning rate schedules
    '/mc/nina/schedule/LearningRateSchedule.php',
    '/mc/nina/schedule/ConstantSchedule.php',
    '/mc/nina/schedule/StepDecaySchedule.php',
    '/mc/nina/schedule/ExponentialSchedule.php',
    '/mc/nina/schedule/PlateauSchedule.php',

    // Weight initializers
    '/mc/nina/initializer/Initializer.php',
    '/mc/nina/initializer/RandomUniformInitializer.php',
    '/mc/nina/initializer/XavierInitializer.php',
    '/mc/nina/initializer/HeInitializer.php',

    // Training algorithms
    '/mc/nina/trainer/Population.php',
    '/mc/nina/trainer/Trainer.php',
    '/mc/nina/BackpropagationTrainer.php',
];

// Load core files
foreach ($coreFiles as $file) {
    $fullPath = NINA_BASE_DIR . $file;
    if (file_exists($fullPath)) {
        require_once $fullPath;
    }
}

// Define commonly used aliases for convenience
if (!class_exists('Network', false)) {
    class_alias('mc\\nina\\Network', 'Network');
}

if (!class_exists('Cell', false)) {
    class_alias('mc\\nina\\Cell', 'Cell');
}

if (!class_exists('Layer', false)) {
    class_alias('mc\\nina\\Layer', 'Layer');
}

// Activation function aliases
if (!class_exists('Sigmoid', false)) {
    class_alias('mc\\nina\\activation\\Sigmoid', 'Sigmoid');
}

if (!class_exists('ReLU', false)) {
    class_alias('mc\\nina\\activation\\ReLU', 'ReLU');
}

if (!class_exists('Tanh', false)) {
    class_alias('mc\\nina\\activation\\Tanh', 'Tanh');
}

// Training algorithm aliases
if (!class_exists('BackpropagationTrainer', false)) {
    class_alias('mc\\nina\\BackpropagationTrainer', 'BackpropagationTrainer');
}

if (!class_exists('MeanSquaredError', false)) {
    class_alias('mc\\nina\\loss\\MeanSquaredError', 'MeanSquaredError');
}

if (!class_exists('CrossEntropy', false)) {
    class_alias('mc\\nina\\loss\\CrossEntropy', 'CrossEntropy');
}

// Optimizer aliases
if (!class_exists('SGD', false)) {
    class_alias('mc\\nina\\optimizer\\SGD', 'SGD');
}

if (!class_exists('Adam', false)) {
    class_alias('mc\\nina\\optimizer\\Adam', 'Adam');
}

if (!class_exists('RMSprop', false)) {
    class_alias('mc\\nina\\optimizer\\RMSprop', 'RMSprop');
}

// Schedule aliases
if (!class_exists('StepDecaySchedule', false)) {
    class_alias('mc\\nina\\schedule\\StepDecaySchedule', 'StepDecaySchedule');
}

if (!class_exists('ExponentialSchedule', false)) {
    class_alias('mc\\nina\\schedule\\ExponentialSchedule', 'ExponentialSchedule');
}

if (!class_exists('PlateauSchedule', false)) {
    class_alias('mc\\nina\\schedule\\PlateauSchedule', 'PlateauSchedule');
}

// Initializer aliases
if (!class_exists('XavierInitializer', false)) {
    class_alias('mc\\nina\\initializer\\XavierInitializer', 'XavierInitializer');
}

if (!class_exists('HeInitializer', false)) {
    class_alias('mc\\nina\\initializer\\HeInitializer', 'HeInitializer');
}

if (!class_exists('RandomUniformInitializer', false)) {
    class_alias('mc\\nina\\initializer\\RandomUniformInitializer', 'RandomUniformInitializer');
}

// Version and info constants
define('NINA_VERSION', '1.0.0');
define('NINA_AUTHOR', 'Mihail Croitor');

// Optional: Display loading message for debugging
if (defined('NINA_DEBUG') && NINA_DEBUG) {
    echo "NiNa Neural Network Library v" . NINA_VERSION . " loaded successfully.\n";
    echo "Autoloader registered for namespace: mc\\nina\\\n";
    echo "Base directory: " . NINA_BASE_DIR . "\n\n";
}