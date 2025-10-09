<?php

use mc\nina\Network;

include_once __DIR__ . "/../src/mc/lib.php";
include_once __DIR__ . "/../src/mc/nina/exception/NinaException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ValidationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/ComputationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/SerializationException.php";
include_once __DIR__ . "/../src/mc/nina/exception/LayerException.php";
include_once __DIR__ . "/../src/mc/nina/exception/NetworkException.php";
include_once __DIR__ . "/../src/mc/nina/exception/TrainingException.php";
include_once __DIR__ . "/../src/mc/nina/activation/ActivationFunction.php";
include_once __DIR__ . "/../src/mc/nina/activation/ReLU.php";
include_once __DIR__ . "/../src/mc/nina/activation/Sigmoid.php";
include_once __DIR__ . "/../src/mc/nina/activation/Tanh.php";
include_once __DIR__ . "/../src/mc/nina/activation/Softmax.php";
include_once __DIR__ . "/../src/mc/nina/cell.php";
include_once __DIR__ . "/../src/mc/nina/layer.php";
include_once __DIR__ . "/../src/mc/nina/network.php";
include_once __DIR__ . "/../src/mc/nina/predictor.php";
include_once __DIR__ . "/../src/mc/nina/trainer.php";

$tests = [
    "0000000000000000" => "00000000000000000000",
    "1000000000000000" => "10000000000000000000",
    "0100000000000000" => "20000000000000000000",
    "0010000000000000" => "30000000000000000000",
    "0001000000000000" => "40000000000000000000",
    "0000100000000000" => "50000000000000000000",
    "0000010000000000" => "60000000000000000000",
    "0000001000000000" => "70000000000000000000",
    "0000000100000000" => "80000000000000000000",
    "0000000010000000" => "90000000000000000000",
];

if (file_exists(__DIR__ . "/sample.json")) {
    $json = file_get_contents(__DIR__ . "/sample.json");
    $network = Network::Load($json);
} else {
    $network = Network::Create("sample", 4, 20);
}

$trainer = new \mc\nina\Trainer($network, 50);

$current_epoch = 0;
$total_epochs = 1000;

while ($current_epoch < $total_epochs) {
    foreach ($tests as $input => $expected) {
        $trainer->Train($input, $expected);
    }
    ++$current_epoch;
}

$best = $trainer->getBestNetwork();
file_put_contents(__DIR__ . "/sample.json", $best->Json());

$predictor = new \mc\nina\Predictor($best);

foreach ($tests as $input => $expected) {
    $result = $predictor->Predict($input);

    echo "Input: {$input} Expected: {$expected} Result: {$result}\n";
}
