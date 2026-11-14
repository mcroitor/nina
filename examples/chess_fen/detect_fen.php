<?php

/**
 * Chess Diagram to FEN Position Detector - CLI Tool
 *
 * Detects the chess position from a diagram image (local PNG file or
 * https://endgame.md/builder/diagram/ URL) and outputs the FEN representation.
 *
 * Usage:
 *   1. Detect FEN from an endgame.md URL:
 *      php examples/chess_fen/detect_fen.php "https://endgame.md/builder/diagram/?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR&size=30&style=alpha&color=black&solid=false&double=false"
 *
 *   2. Detect FEN from a local image file:
 *      php examples/chess_fen/detect_fen.php examples/chess_fen/diagrams/initial_position.png
 *
 *   3. Generate a diagram from a FEN via endgame.md and verify round-trip detection:
 *      php examples/chess_fen/detect_fen.php --fen "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"
 *
 *   4. Quiet mode (outputs ONLY the FEN string for scripting):
 *      php examples/chess_fen/detect_fen.php --quiet examples/chess_fen/diagrams/initial_position.png
 */

require_once __DIR__ . '/chess_common.php';

use mc\nina\Network;

$modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'chess_fen_model.json';
if (!file_exists($modelPath)) {
    fwrite(STDERR, "Error: Model file not found at {$modelPath}\n");
    fwrite(STDERR, "Please run 'php examples/chess_fen/train_chess_fen.php' first.\n");
    exit(1);
}

$network = Network::Load((string)file_get_contents($modelPath));

// Parse CLI arguments
$quiet = false;
$expectedFen = null;
$source = null;

for ($i = 1; $i < $argc; $i++) {
    $arg = $argv[$i];
    if ($arg === '--help' || $arg === '-h') {
        printChessHelp();
        exit(0);
    } elseif ($arg === '--quiet' || $arg === '-q') {
        $quiet = true;
    } elseif ($arg === '--fen' && isset($argv[$i + 1])) {
        $expectedFen = explode(' ', trim($argv[++$i]))[0];
        $source = buildEndgameDiagramUrl($expectedFen, 30, 'alpha', 'black', false, false);
    } else {
        $source = $arg;
    }
}

if ($source === null) {
    // Default demo: run on initial position diagram (local cache or URL)
    $defaultFile = __DIR__ . DIRECTORY_SEPARATOR . 'diagrams' . DIRECTORY_SEPARATOR . 'initial_position.png';
    if (file_exists($defaultFile)) {
        $source = $defaultFile;
        $expectedFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
    } else {
        $expectedFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
        $source = buildEndgameDiagramUrl($expectedFen, 30, 'alpha', 'black', false, false);
    }
}

// If source is an endgame.md URL with ?fen=..., extract expected FEN for verification display
if ($expectedFen === null && preg_match('/[?&]fen=([^&]+)/i', $source, $m)) {
    $expectedFen = explode(' ', rawurldecode($m[1]))[0];
}

try {
    $t0 = microtime(true);
    $img = loadChessDiagramImage($source);
    $result = detectPositionFromDiagram($network, $img, 30);
    $elapsedMs = (microtime(true) - $t0) * 1000.0;

    if ($quiet) {
        echo $result['fen'] . PHP_EOL;
        exit(0);
    }

    echo "\n=========================================================================\n";
    echo "   ♟️  NiNa Chess Diagram → FEN Position Detector\n";
    echo "=========================================================================\n";
    echo "Source: " . $source . "\n";
    echo "Image:  " . imagesx($img) . "×" . imagesy($img) . " px\n\n";

    echo "Detected Board Position:\n";
    echo renderChessboardAscii($result['grid']);
    echo "\n";

    echo "🎯 Detected FEN:   " . $result['fen'] . "\n";
    if ($expectedFen !== null) {
        $match = ($result['fen'] === $expectedFen);
        echo "📋 Expected FEN:   " . $expectedFen . "\n";
        echo "🔎 Verification:   " . ($match ? "✅ 100% EXACT MATCH" : "❌ MISMATCH") . "\n";
    }
    echo sprintf(
        "📊 Confidence:     mean=%.1f%%, min=%.1f%% (64 squares classified in %.1f ms)\n\n",
        $result['mean_confidence'] * 100,
        $result['min_confidence'] * 100,
        $elapsedMs
    );

} catch (\Throwable $e) {
    fwrite(STDERR, "❌ Error: " . $e->getMessage() . "\n");
    exit(1);
}

function printChessHelp(): void
{
    echo <<<HELP

NiNa Chess Diagram to FEN Detector
==================================

Usage:
  php examples/chess_fen/detect_fen.php <image_file_or_url>
  php examples/chess_fen/detect_fen.php --fen "<fen_string>"
  php examples/chess_fen/detect_fen.php --quiet <image_file_or_url>

Examples:
  php examples/chess_fen/detect_fen.php "https://endgame.md/builder/diagram/?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR&size=30&style=alpha&color=black&solid=false&double=false"
  php examples/chess_fen/detect_fen.php examples/chess_fen/diagrams/sicilian_najdorf.png
  php examples/chess_fen/detect_fen.php --fen "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R"

HELP;
}
