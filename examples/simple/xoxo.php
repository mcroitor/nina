<?php

/**
 * Tic-Tac-Toe (XOXO) - Neural Network Interactive Game & Evaluator
 * 
 * Powered by NiNa Neural Network Library.
 * 
 * Usage:
 *   Interactive Game:
 *     php examples/simple/xoxo.php
 * 
 *   One-shot Board Evaluation:
 *     php examples/simple/xoxo.php <board-string>
 *     Example: php examples/simple/xoxo.php 100200000
 *     Example: php examples/simple/xoxo.php X..O.....
 */

require_once __DIR__ . "/../../src/autoload.php";

use mc\nina\Network;

$WINNING_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
    [0, 4, 8], [2, 4, 6]             // Diagonals
];

// Load neural network model
$modelFile = __DIR__ . '/xoxo_model.json';
if (!file_exists($modelFile)) {
    fwrite(STDERR, "Error: Network model file not found: {$modelFile}\n");
    fwrite(STDERR, "Please run 'php examples/simple/train_xoxo.php' to train the model first.\n");
    exit(1);
}

$network = Network::Load(file_get_contents($modelFile));

// Parse CLI arguments
if ($argc > 1) {
    $arg = trim($argv[1]);
    if ($arg === '--help' || $arg === '-h') {
        printHelp();
        exit(0);
    }
    evaluateBoardCli($arg, $network);
    exit(0);
}

// Default mode: Interactive game
playInteractiveGame($network);

/**
 * Renders the 3x3 Tic-Tac-Toe board with position guides for empty cells
 */
function renderBoard(array $board): string
{
    $display = [];
    for ($i = 0; $i < 9; $i++) {
        if ($board[$i] !== ' ') {
            $display[$i] = $board[$i];
        } else {
            // Show position number 1-9 for easy input
            $display[$i] = (string)($i + 1);
        }
    }

    $lines = [];
    $lines[] = "     {$display[0]} | {$display[1]} | {$display[2]} ";
    $lines[] = "    ---+---+---";
    $lines[] = "     {$display[3]} | {$display[4]} | {$display[5]} ";
    $lines[] = "    ---+---+---";
    $lines[] = "     {$display[6]} | {$display[7]} | {$display[8]} ";

    return implode("\n", $lines) . "\n";
}

/**
 * Checks for a winner or completed line
 * 
 * @return array|null Returns ['winner' => 'X'|'O', 'line' => [p1, p2, p3]] or null
 */
function checkWinner(array $board): ?array
{
    global $WINNING_LINES;

    foreach ($WINNING_LINES as $line) {
        $m0 = $board[$line[0]];
        $m1 = $board[$line[1]];
        $m2 = $board[$line[2]];

        if ($m0 !== ' ' && $m0 === $m1 && $m1 === $m2) {
            return ['winner' => $m0, 'line' => $line];
        }
    }

    return null;
}

/**
 * Checks if the board is completely full (draw)
 */
function isBoardFull(array $board): bool
{
    return !in_array(' ', $board, true);
}

/**
 * Computes AI's next move using NiNa neural network evaluation
 */
function getAiMove(array $board, Network $network, string $aiMark, string $humanMark): int
{
    global $WINNING_LINES;

    // 1. Tactical Win: If AI can complete a line, take it immediately
    foreach ($WINNING_LINES as $line) {
        $marks = [$board[$line[0]], $board[$line[1]], $board[$line[2]]];
        if (count(array_keys($marks, $aiMark, true)) === 2 && count(array_keys($marks, ' ', true)) === 1) {
            foreach ($line as $pos) {
                if ($board[$pos] === ' ') {
                    return $pos;
                }
            }
        }
    }

    // 2. Tactical Block: If Human is about to win, block the line
    foreach ($WINNING_LINES as $line) {
        $marks = [$board[$line[0]], $board[$line[1]], $board[$line[2]]];
        if (count(array_keys($marks, $humanMark, true)) === 2 && count(array_keys($marks, ' ', true)) === 1) {
            foreach ($line as $pos) {
                if ($board[$pos] === ' ') {
                    return $pos;
                }
            }
        }
    }

    // 3. Neural Network Preference:
    // Convert board to normalized numeric input: 0.0 for empty, 0.5 for X, 1.0 for O
    $numeric = [];
    for ($i = 0; $i < 9; $i++) {
        if ($board[$i] === 'X') {
            $numeric[] = 0.5;
        } elseif ($board[$i] === 'O') {
            $numeric[] = 1.0;
        } else {
            $numeric[] = 0.0;
        }
    }

    $scores = $network->Calculate($numeric);

    // Pick available cell with highest neural network activation
    $bestPos = -1;
    $bestScore = -INF;
    for ($i = 0; $i < 9; $i++) {
        if ($board[$i] === ' ' && $scores[$i] > $bestScore) {
            $bestScore = $scores[$i];
            $bestPos = $i;
        }
    }

    if ($bestPos !== -1) {
        return $bestPos;
    }

    // 4. Fallback: Center or first available
    if ($board[4] === ' ') {
        return 4;
    }

    for ($i = 0; $i < 9; $i++) {
        if ($board[$i] === ' ') {
            return $i;
        }
    }

    return 0;
}

/**
 * Main interactive game loop
 */
function playInteractiveGame(Network $network): void
{
    echo "\n";
    echo "=========================================\n";
    echo "      TIC-TAC-TOE (XOXO) vs NiNa AI      \n";
    echo "=========================================\n";
    echo "Play against the trained neural network!\n";
    echo "Grid positions are numbered 1 through 9.\n\n";

    while (true) {
        $board = array_fill(0, 9, ' ');

        // Choose player mark
        echo "Play as X (first move) or O (second move)? [X/o]: ";
        $choice = trim((string)fgets(STDIN));
        if (feof(STDIN)) {
            echo "\nGoodbye!\n";
            break;
        }

        $humanMark = (strtoupper($choice) === 'O') ? 'O' : 'X';
        $aiMark = ($humanMark === 'X') ? 'O' : 'X';
        $currentTurn = 'X';

        echo "\nYou are '{$humanMark}'. AI is '{$aiMark}'.\n";
        echo ($humanMark === 'X') ? "You go first!\n" : "AI goes first!\n";

        // Game loop
        while (true) {
            echo "\n" . renderBoard($board);

            // Check if game ended
            $winResult = checkWinner($board);
            if ($winResult !== null) {
                if ($winResult['winner'] === $humanMark) {
                    echo "🎉 Congratulations! You won!\n";
                } else {
                    echo "🤖 AI wins! Better luck next time.\n";
                }
                break;
            }

            if (isBoardFull($board)) {
                echo "🤝 It's a draw! Well played.\n";
                break;
            }

            // Execute move
            if ($currentTurn === $humanMark) {
                // Human Turn
                while (true) {
                    echo "Your turn ({$humanMark}). Enter position (1-9) or 'q' to quit: ";
                    $input = trim((string)fgets(STDIN));

                    if (feof(STDIN) || strtolower($input) === 'q' || strtolower($input) === 'quit') {
                        echo "\nGame aborted. Goodbye!\n";
                        return;
                    }

                    if (!ctype_digit($input) || (int)$input < 1 || (int)$input > 9) {
                        echo "⚠️  Invalid move. Please enter a number between 1 and 9.\n";
                        continue;
                    }

                    $pos = (int)$input - 1;
                    if ($board[$pos] !== ' ') {
                        echo "⚠️  Position " . ($pos + 1) . " is already occupied. Choose another cell.\n";
                        continue;
                    }

                    $board[$pos] = $humanMark;
                    break;
                }
                $currentTurn = $aiMark;
            } else {
                // AI Turn
                echo "AI ({$aiMark}) is evaluating the board...\n";
                $aiPos = getAiMove($board, $network, $aiMark, $humanMark);
                $board[$aiPos] = $aiMark;
                echo "🤖 AI places '{$aiMark}' at position " . ($aiPos + 1) . ".\n";
                $currentTurn = $humanMark;
            }
        }

        // Ask for rematch
        echo "\nPlay another game? [y/N]: ";
        $again = trim((string)fgets(STDIN));
        if (feof(STDIN) || strtolower($again) !== 'y') {
            echo "Thanks for playing!\n\n";
            break;
        }
    }
}

/**
 * One-shot CLI evaluation mode
 */
function evaluateBoardCli(string $inputString, Network $network): void
{
    $inputString = trim($inputString);

    if (strlen($inputString) !== 9) {
        fwrite(STDERR, "Error: Board string must be exactly 9 characters long.\n");
        fwrite(STDERR, "Got: '{$inputString}' (" . strlen($inputString) . " characters)\n");
        exit(1);
    }

    // Parse input string
    $board = [];
    $xCount = 0;
    $oCount = 0;

    for ($i = 0; $i < 9; $i++) {
        $c = $inputString[$i];
        if ($c === '1' || strtoupper($c) === 'X') {
            $board[$i] = 'X';
            $xCount++;
        } elseif ($c === '2' || strtoupper($c) === 'O') {
            $board[$i] = 'O';
            $oCount++;
        } elseif ($c === '0' || $c === '.' || $c === ' ' || $c === '_') {
            $board[$i] = ' ';
        } else {
            fwrite(STDERR, "Error: Invalid character '{$c}' at position " . ($i + 1) . ".\n");
            fwrite(STDERR, "Allowed characters: 0, 1, 2 or ., X, O\n");
            exit(1);
        }
    }

    // Determine whose turn it is: X goes first, so if xCount == oCount, it's X's turn, else O
    $turn = ($xCount <= $oCount) ? 'X' : 'O';
    $opponent = ($turn === 'X') ? 'O' : 'X';

    echo "\n=== Board Evaluation ===\n";
    echo "Current Board State:\n";
    echo renderBoard($board);

    $winResult = checkWinner($board);
    if ($winResult !== null) {
        echo "Game is already over: '{$winResult['winner']}' has won!\n\n";
        return;
    }

    if (isBoardFull($board)) {
        echo "Game is already over: Board is full (Draw)!\n\n";
        return;
    }

    echo "Evaluating next move for '{$turn}'...\n";
    $move = getAiMove($board, $network, $turn, $opponent);

    $row = (int)floor($move / 3) + 1;
    $col = ($move % 3) + 1;
    echo "Recommended move: Position " . ($move + 1) . " (Row {$row}, Col {$col})\n\n";

    $boardAfter = $board;
    $boardAfter[$move] = $turn;

    echo "Board After Move:\n";
    echo renderBoard($boardAfter);

    // Format output string
    $outStr = '';
    for ($i = 0; $i < 9; $i++) {
        if ($boardAfter[$i] === 'X') $outStr .= '1';
        elseif ($boardAfter[$i] === 'O') $outStr .= '2';
        else $outStr .= '0';
    }
    echo "Output state: {$outStr}\n\n";
}

/**
 * Prints help message
 */
function printHelp(): void
{
    echo <<<HELP

NiNa Tic-Tac-Toe (XOXO) Game & Evaluator
========================================

Usage:
  Interactive Mode:
    php examples/simple/xoxo.php

  One-Shot Board Evaluation:
    php examples/simple/xoxo.php <board-string>

Board String Formats (9 characters):
  - Numeric: 0 = empty, 1 = X, 2 = O (e.g. "100200000")
  - Symbols: . = empty, X = X, O = O (e.g. "X..O.....")

Grid Coordinates:
   1 | 2 | 3 
  ---+---+---
   4 | 5 | 6 
  ---+---+---
   7 | 8 | 9 

Examples:
  php examples/simple/xoxo.php 000000000
  php examples/simple/xoxo.php X.O..X...

HELP;
}
