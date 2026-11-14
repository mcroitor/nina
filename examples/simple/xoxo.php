<?php
/**
 * model plays Tic-Tac-Toe
 * input is an array of 9 values: 1 for X, 2 for O, or 0 for empty
 * output is an array of 9 values with the next move made by 'X'
 * 
 * Example usage:
 * 
 * ```bash
 * php examples/simple/xoxo.php <input-string>
 * ```
 */

include __DIR__ . "/../../src/autoload.php";

use mc\nina\Network;
use mc\nina\Predictor;

if ($argc < 2) {
    echo "Usage: php xoxo.php <input-string>\n";
    echo "Example: php xoxo.php 1002000200\n";
    exit(1);
}

$inputString = $argv[1];
if (strlen($inputString) != 9 || !preg_match('/^[012]{9}$/', $inputString)) {
    echo "Input string must be 9 characters long, containing only 0, 1, or 2.\n";
    exit(1);
}

$inputArray = array_map('intval', str_split($inputString));

// Load pre-trained network
$networkFile = __DIR__ . '/xoxo_model.json';

if (!file_exists($networkFile)) {
    echo "Network model file not found: $networkFile\n";
    exit(1);
}

$networkJson = file_get_contents($networkFile);
$network = Network::Load($networkJson);

$predictor = new Predictor($network);
$outputString = $predictor->Predict($inputString);
echo "Input:  $inputString\n";
echo "Output: $outputString\n";
