<?php

include __DIR__ . "/../../src/autoload.php";

use mc\nina\activation\Sigmoid;
use mc\nina\Network;
use mc\nina\Trainer;
use mc\nina\Predictor;

const __DEBUG__ = true;

function prepareString(array $data): string
{
    return implode('', array_map(fn($v) => chr(intval($v * 255)), $data));
}

$TRAIN_DATA = [
    [
        'input' => [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ],
        'target' => [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        'target' => [
            1,
            2,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        'target' => [
            1,
            0,
            2,
            1,
            0,
            0,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
        ],
        'target' => [
            1,
            1,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
        ],
        'target' => [
            1,
            0,
            0,
            1,
            2,
            0,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
        ],
        'target' => [
            1,
            0,
            1,
            0,
            0,
            2,
            0,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
        ],
        'target' => [
            1,
            1,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
        ],
        'target' => [
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            2,
            0,
        ]
    ],
    [
        'input' => [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
        ],
        'target' => [
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            2,
        ]
    ],
    [
        'input' => [
            1,
            2,
            0,
            1,
            0,
            0,
            2,
            0,
            0,
        ],
        'target' => [
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            0,
            0,
        ]
    ],
    [
        'input' => [
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            0,
            2,
        ],
        'target' => [
            1,
            2,
            0,
            1,
            1,
            1,
            2,
            0,
            2,
        ]
    ],
    [
        'input' => [
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            2,
            0,
        ],
        'target' => [
            1,
            2,
            0,
            1,
            1,
            0,
            2,
            2,
            1,
        ]
    ],
];

// network architecture
$NET_NR_LAYERS = 3;
$NET_LAYER_SIZE = 9;
$NET_ACTIVATION = "Sigmoid";
$NET_FILE = "xoxo_model.json";
// training parameters
$MAX_GENERATIONS = 5000;
$MUTATION_RATE = 10; // in percentage
$TARGET_FITNESS = 0.95;
$POPULATION_SIZE = 100;

echo "🎮 XOXO Game AI Training\n";
echo "========================\n";
echo "Training data samples: " . count($TRAIN_DATA) . "\n";
echo "Network architecture: $NET_NR_LAYERS layers × $NET_LAYER_SIZE neurons\n";
echo "Population size: $POPULATION_SIZE\n";
echo "Mutation rate: $MUTATION_RATE%\n\n";

try {
    if (file_exists($NET_FILE)) {
        echo "Loading existing network from $NET_FILE...\n";
        $networkJson = file_get_contents($NET_FILE);
        $network = Network::Load($networkJson);
        echo "Network loaded successfully.\n\n";
    } else {
        echo "Creating new network...\n";
        $network = Network::Create(
            "XOXO",
            $NET_NR_LAYERS,
            $NET_LAYER_SIZE,
            new Sigmoid()
        );
        echo "New network created.\n\n";
    }

    $trainer = new Trainer($network, $POPULATION_SIZE, $MUTATION_RATE);

    // train network
    echo "Starting training with $POPULATION_SIZE networks for max $MAX_GENERATIONS generations...\n";
    echo "Target fitness: " . ($TARGET_FITNESS * 100) . "%\n\n";

    foreach ($TRAIN_DATA as $sample) {
        for ($i = 0; $i < $MAX_GENERATIONS; $i++) {
            // Train on all samples in this generation
            $trainer->Train(prepareString($sample['input']), prepareString($sample['target']));
        }
        // Evaluate fitness less frequently for performance
        if ($i % 10 == 0 || $i == $MAX_GENERATIONS - 1) {
            $bestNetwork = $trainer->getBestNetwork();
            $predictor = new Predictor($bestNetwork);
            $correctPredictions = 0;

            // Check fitness on all training samples
            foreach ($TRAIN_DATA as $sample) {
                $inputStr = prepareString($sample['input']);
                $expectedStr = prepareString($sample['target']);
                $outputStr = $predictor->Predict($inputStr);

                if ($outputStr === $expectedStr) {
                    $correctPredictions++;
                }
            }

            $fitness = $correctPredictions / count($TRAIN_DATA);

            // Check if target reached
            if ($fitness >= $TARGET_FITNESS) {
                echo "🎯 Target fitness reached: " . round($fitness * 100, 2) . "% at generation $i\n";
                break;
            }

            // Progress reporting
            if ($i % 100 == 0) {
                echo "Generation $i: fitness = " . round($fitness * 100, 2) . "%\n";
            }
        }
    }
    // Get final best network
    $bestNetwork = $trainer->getBestNetwork();

    // Final evaluation
    echo "\n🧪 Final Evaluation:\n";
    echo "====================\n";
    $predictor = new Predictor($bestNetwork);
    $totalCorrect = 0;

    foreach ($TRAIN_DATA as $index => $sample) {
        $inputStr = prepareString($sample['input']);
        $expectedStr = prepareString($sample['target']);
        $outputStr = $predictor->Predict($inputStr);

        $isCorrect = $outputStr === $expectedStr;
        if ($isCorrect) {
            $totalCorrect++;
        }

        echo "Sample " . ($index + 1) . ": " . ($isCorrect ? "✅ PASS" : "❌ FAIL") . "\n";
        echo "  Input:    [" . implode(', ', $sample['input']) . "]\n";
        echo "  Expected: [" . implode(', ', $sample['target']) . "]\n";

        // Convert output back to array for display
        $outputArray = array_map(fn($char) => ord($char) / 255.0, str_split($outputStr));
        $outputArrayRounded = array_map(fn($val) => round($val, 3), $outputArray);
        echo "  Got:      [" . implode(', ', $outputArrayRounded) . "]\n\n";
    }

    $finalAccuracy = $totalCorrect / count($TRAIN_DATA);
    echo "🎯 Final Accuracy: " . round($finalAccuracy * 100, 2) . "% ($totalCorrect/" . count($TRAIN_DATA) . ")\n\n";

        // save network
    ;
    $networkJson = $bestNetwork->Json();

    if (file_put_contents($NET_FILE, $networkJson) !== false) {
        echo "💾 Network saved to: $NET_FILE\n";
        echo "📊 Model size: " . number_format(strlen($networkJson)) . " bytes\n";
    } else {
        echo "⚠️  Warning: Failed to save network to $NET_FILE\n";
    }

    echo "\n🎉 Training completed successfully!\n";
} catch (Exception $e) {
    echo "\n❌ Error during training: " . $e->getMessage() . "\n";
    echo "📍 File: " . $e->getFile() . " line " . $e->getLine() . "\n";

    if (method_exists($e, 'getTraceAsString')) {
        echo "\n📋 Stack trace:\n" . $e->getTraceAsString() . "\n";
    }

    exit(1);
}
