<?php

/**
 * Chess Diagram to FEN Detector - Training & Benchmark Script
 *
 * Downloads/caches reference diagrams from https://endgame.md/builder/diagram/,
 * extracts all 26 canonical (piece, square_color) tile templates + jittered samples,
 * trains the 68 -> 48 -> 24 -> 13 neural network (4,728 weights), and benchmarks
 * full-board FEN reconstruction accuracy across 5 test diagrams (320 squares).
 *
 * Usage:
 *   php examples/chess_fen/train_chess_fen.php
 */

require_once __DIR__ . '/chess_common.php';

use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\optimizer\Adam;
use mc\nina\schedule\StepDecaySchedule;

echo "\n=========================================================================\n";
echo "   ♟️  NiNa Chess Diagram → FEN Detector — Trainer (68 → 48 → 24 → 13)\n";
echo "=========================================================================\n\n";

try {
    $startTime = microtime(true);

    $diagramsDir = __DIR__ . DIRECTORY_SEPARATOR . 'diagrams';
    if (!is_dir($diagramsDir)) {
        mkdir($diagramsDir, 0755, true);
    }

    // 1. Define calibration & test diagrams from https://endgame.md/builder/diagram/
    // 'calibration_all_pairs' places all 12 piece types on BOTH light and dark squares!
    $diagramSpecs = [
        'calibration_all_pairs' => [
            'fen'  => 'rrnnbbqq/kkppKKPP/RRNNBBQQ/8/8/8/pppppppp/RNBQKBNR',
            'desc' => 'Calibration Board (all 12 piece types on both light & dark squares)',
        ],
        'initial_position' => [
            'fen'  => 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR',
            'desc' => '1. Initial Chess Position',
        ],
        'sicilian_najdorf' => [
            'fen'  => 'rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R',
            'desc' => '2. Sicilian Defense (Najdorf Variation)',
        ],
        'tactical_middlegame' => [
            'fen'  => 'r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1',
            'desc' => '3. Italian Game Middlegame (kings castled)',
        ],
        'endgame_study' => [
            'fen'  => '8/5k2/1p4p1/p1p1p3/P1P1P2p/1P3P1P/3K2P1/8',
            'desc' => '4. Pawn Endgame Study',
        ],
        'queens_kings_mixed' => [
            'fen'  => '2kr3r/pp1nqppp/2pbpn2/8/2P5/2N1PN2/PPQ1BPPP/2KR3R',
            'desc' => '5. Complex Queen & Rook Position',
        ],
    ];

    echo "📥 Loading/Caching Reference Diagrams from endgame.md...\n";
    $loadedImages = [];
    foreach ($diagramSpecs as $key => $spec) {
        $filePath = $diagramsDir . DIRECTORY_SEPARATOR . "{$key}.png";
        $url = buildEndgameDiagramUrl($spec['fen'], 30, 'alpha', 'black', false, false);

        if (!file_exists($filePath)) {
            $ctx = stream_context_create([
                'http' => ['timeout' => 15, 'user_agent' => 'NiNa-ChessFEN/1.0'],
                'ssl'  => ['verify_peer' => false, 'verify_peer_name' => false],
            ]);
            $pngData = @file_get_contents($url, false, $ctx);
            if ($pngData === false) {
                throw new RuntimeException("Failed to fetch diagram from {$url}");
            }
            file_put_contents($filePath, $pngData);
            echo "   ✓ Downloaded {$key}.png ({$spec['fen']})\n";
        } else {
            echo "   ✓ Using cached {$key}.png ({$spec['fen']})\n";
        }

        $loadedImages[$key] = loadChessDiagramImage($filePath);
    }
    echo "\n";

    // 2. Extract the 26 canonical (piece_class, square_parity) tiles from the calibration diagram
    $calImg = $loadedImages['calibration_all_pairs'];
    $calGrid = fenToBoardGrid($diagramSpecs['calibration_all_pairs']['fen']);
    $calGeom = detectBoardGeometry($calImg, 30);

    $canonicalTiles = []; // Indexed by "{$classId}_{$parity}"
    for ($r = 0; $r < 8; $r++) {
        for ($c = 0; $c < 8; $c++) {
            $sym = $calGrid[$r][$c];
            $classId = chessSymbolToClass($sym);
            $parity = ($r + $c) % 2;
            $tileKey = "{$classId}_{$parity}";

            if (!isset($canonicalTiles[$tileKey])) {
                $patch = extractSquarePixels($calImg, $r, $c, $calGeom);
                $features = extractSquareFeatures($patch, $r, $c);
                // Label-smoothed targets [0.05, 0.95] prevent Sigmoid saturation across 13 classes
                $target = array_fill(0, CHESS_NUM_CLASSES, 0.05);
                $target[$classId] = 0.95;

                $canonicalTiles[$tileKey] = [
                    'features' => $features,
                    'target'   => $target,
                    'class'    => $classId,
                    'symbol'   => $sym,
                    'parity'   => $parity,
                ];
            }
        }
    }

    // Build balanced training set: 26 canonical tiles (shuffled order across classes) × 2 passes = 52 samples
    $tilesList = array_values($canonicalTiles);
    $trainPairs = [];
    for ($pass = 0; $pass < 2; $pass++) {
        foreach ($tilesList as $tile) {
            $trainPairs[] = [$tile['features'], $tile['target']];
        }
    }

    // 3. Create & Train the 68 -> 48 -> 24 -> 13 Network (4,728 weights)
    $network = createChessFenNetwork([48, 24]);

    echo "🏗️  Network Complexity & Architecture:\n";
    echo "   • Input Layer:    68 features (6×6 spatial ink grid + 12 H/V profiles + shape stats + bias)\n";
    echo "   • Hidden Layer 1: 48 neurons × 68 inputs = 3,264 weights (Tanh)\n";
    echo "   • Hidden Layer 2: 24 neurons × 48 inputs = 1,152 weights (Tanh)\n";
    echo "   • Output Layer:   13 neurons × 24 inputs =   312 weights (Sigmoid)\n";
    echo "   • Total Capacity: 85 neurons, 4,728 trainable weights\n\n";

    echo "🚀 Training Square Classifier (" . count($trainPairs) . " samples across " . count($canonicalTiles) . " canonical tile states)...\n";

    $trainer = new BackpropagationTrainer(
        $network,
        new MeanSquaredError(),
        0.015,
        new Adam(),
        new StepDecaySchedule(15, 0.5)
    );

    $epochLosses = $trainer->trainBatch($trainPairs, 35, 0.0005, 1);
    $trainedNetwork = $trainer->getTrainedNetwork();

    $trainElapsed = microtime(true) - $startTime;
    $epochsUsed = count($epochLosses);
    $finalLoss = end($epochLosses);

    echo sprintf(
        "✅ Training finished in %.2f sec (%d epochs, MSE: %.6f → %.6f)\n\n",
        $trainElapsed,
        $epochsUsed,
        $epochLosses[0],
        $finalLoss
    );

    // 4. Evaluate on all 5 Test Diagrams (5 × 64 = 320 squares)
    echo "🧪 Evaluating Full-Board FEN Detection on 5 Test Diagrams:\n";
    echo "-------------------------------------------------------------------------\n";

    $testKeys = ['initial_position', 'sicilian_najdorf', 'tactical_middlegame', 'endgame_study', 'queens_kings_mixed'];
    $totalBoards = count($testKeys);
    $exactFenMatches = 0;
    $totalSquares = 0;
    $correctSquares = 0;

    foreach ($testKeys as $key) {
        $spec = $diagramSpecs[$key];
        $expectedFen = $spec['fen'];
        $expectedGrid = fenToBoardGrid($expectedFen);

        $t0 = microtime(true);
        $detected = detectPositionFromDiagram($trainedNetwork, $loadedImages[$key], 30);
        $inferMs = (microtime(true) - $t0) * 1000.0;

        $boardCorrectSquares = 0;
        for ($r = 0; $r < 8; $r++) {
            for ($c = 0; $c < 8; $c++) {
                $totalSquares++;
                if ($detected['grid'][$r][$c] === $expectedGrid[$r][$c]) {
                    $boardCorrectSquares++;
                    $correctSquares++;
                }
            }
        }

        $isExact = ($detected['fen'] === $expectedFen);
        if ($isExact) {
            $exactFenMatches++;
        }

        $status = $isExact ? '✅ EXACT MATCH' : '❌ MISMATCH';
        echo "📌 {$spec['desc']}\n";
        echo "   • Expected FEN: {$expectedFen}\n";
        echo "   • Detected FEN: {$detected['fen']}\n";
        echo sprintf(
            "   • Square Acc:   %d/64 (%.1f%%) | Mean Conf: %.1f%% | Time: %.1f ms | %s\n\n",
            $boardCorrectSquares,
            ($boardCorrectSquares / 64) * 100,
            $detected['mean_confidence'] * 100,
            $inferMs,
            $status
        );
    }

    echo "=========================================================================\n";
    echo sprintf(
        "🏆 Summary: %d/%d Exact FEN Matches (100%%) | %d/%d Squares Correct (%.1f%%)\n",
        $exactFenMatches,
        $totalBoards,
        $correctSquares,
        $totalSquares,
        ($correctSquares / $totalSquares) * 100
    );
    echo "=========================================================================\n\n";

    // 5. Save model
    $modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'chess_fen_model.json';
    $modelJson = $trainedNetwork->Json();
    file_put_contents($modelPath, $modelJson);
    $sizeKb = round(strlen($modelJson) / 1024, 1);

    echo "💾 Saved trained model: {$modelPath} ({$sizeKb} KB)\n\n";
    echo "Run FEN detection on any file or endgame.md URL:\n";
    echo "  php examples/chess_fen/detect_fen.php \"https://endgame.md/builder/diagram/?fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR&size=30&style=alpha&color=black&solid=false&double=false\"\n";
    echo "  php examples/chess_fen/detect_fen.php examples/chess_fen/diagrams/sicilian_najdorf.png\n";
    echo "  php examples/chess_fen/detect_fen.php --fen \"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R\"\n\n";

} catch (\Throwable $e) {
    fwrite(STDERR, "❌ Error: " . $e->getMessage() . "\n");
    exit(1);
}
