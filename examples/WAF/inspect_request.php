<?php

/**
 * NiNa Web Application Firewall (WAF) - Request Inspector CLI
 *
 * Inspects HTTP query strings, form fields, headers, or JSON payloads using
 * the trained 24 -> 32 -> 16 -> 5 neural network model.
 *
 * Usage:
 *   1. Live showcase on diverse benign & attack payloads:
 *      php examples/WAF/inspect_request.php
 *
 *   2. Inspect a custom payload string:
 *      php examples/WAF/inspect_request.php "<payload>"
 *
 *   3. Inspect a batch of payloads from a JSON file:
 *      php examples/WAF/inspect_request.php --file examples/WAF/sample_payloads.json
 */

require_once __DIR__ . '/waf_common.php';

use mc\nina\Network;

$modelPath = __DIR__ . DIRECTORY_SEPARATOR . 'waf_model.json';
if (!file_exists($modelPath)) {
    fwrite(STDERR, "Error: WAF model file not found at {$modelPath}\n");
    fwrite(STDERR, "Please run 'php examples/WAF/train_waf.php' first.\n");
    exit(1);
}

$network = Network::Load((string)file_get_contents($modelPath));

if ($argc > 1) {
    $arg1 = $argv[1];

    if ($arg1 === '--help' || $arg1 === '-h') {
        printWafHelp();
        exit(0);
    }

    if ($arg1 === '--file' || $arg1 === '-f') {
        if (!isset($argv[2])) {
            fwrite(STDERR, "Error: Missing file path after --file\n");
            exit(1);
        }
        inspectBatchFile($network, $argv[2]);
        exit(0);
    }

    // Single payload inspection
    $payload = implode(' ', array_slice($argv, 1));
    printDetailedInspection($network, $payload);
    exit(0);
}

// Default: run live showcase
runWafShowcase($network);

/**
 * Prints a detailed neural WAF report for a single HTTP payload.
 */
function printDetailedInspection(Network $network, string $payload): void
{
    $analysis = classifyWafPayload($network, $payload);
    $classInfo = getWafClassInfo();
    $featureNames = getWafFeatureNames();

    echo "\n======================================================================\n";
    echo "   🛡️  NiNa Neural WAF — Deep Packet / Payload Inspection\n";
    echo "======================================================================\n";
    echo "Payload:  " . $payload . "\n";
    echo "Length:   " . strlen($payload) . " bytes\n\n";

    // Display active extracted features
    echo "Active Extracted Signals (Top Features > 0.05):\n";
    echo "----------------------------------------------------------------------\n";
    for ($i = 0; $i < WAF_INPUT_SIZE - 1; $i++) {
        $val = $analysis['features'][$i];
        if ($val >= 0.05) {
            $barLen = (int)round($val * 20);
            $bar = str_repeat('▪', $barLen) . str_repeat('·', 20 - $barLen);
            echo sprintf("  f[%02d] %-24s |%s| %.2f\n", $i, $featureNames[$i], $bar, $val);
        }
    }
    echo "----------------------------------------------------------------------\n\n";

    // Display class probability distribution
    echo "Threat Class Distribution:\n";
    echo "----------------------------------------------------------------------\n";
    for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
        $info = $classInfo[$c];
        $prob = $analysis['probabilities'][$c];
        $raw = $analysis['raw_outputs'][$c];
        $barLen = (int)round(max(0.0, min(1.0, $prob)) * 25);
        $bar = str_repeat('█', $barLen) . str_repeat('░', 25 - $barLen);
        $marker = ($c === $analysis['predicted_class']) ? ' ◄ MATCH' : '';
        echo sprintf(
            "  [%d] %-7s |%s| %5.1f%% (act: %.3f)%s\n",
            $c,
            $info['code'],
            $bar,
            $prob * 100,
            $raw,
            $marker
        );
    }
    echo "----------------------------------------------------------------------\n\n";

    $icon = ($analysis['predicted_class'] === WAF_CLASS_BENIGN) ? '✅' : '🚨';
    echo "{$icon} WAF VERDICT:\n";
    echo "   • Classification: {$analysis['class_code']} ({$analysis['class_name']})\n";
    echo "   • Severity:       {$analysis['severity']}\n";
    echo sprintf("   • Threat Score:   %.1f%%\n", $analysis['threat_score'] * 100);
    echo "   • Action:         {$analysis['action']}\n\n";
}

/**
 * Runs a live showcase on unseen test requests and displays a summary table + deep dive.
 */
function runWafShowcase(Network $network): void
{
    $corpus = getWafCorpus();
    $testPayloads = $corpus['test'];
    $classInfo = getWafClassInfo();

    echo "\n====================================================================================\n";
    echo "   🛡️  NiNa Neural WAF — Live Inspection Showcase (Architecture: 24 → 32 → 16 → 5)\n";
    echo "====================================================================================\n\n";

    echo " Expected | Predicted | Threat% | WAF Action            | HTTP Request Payload\n";
    echo "----------+-----------+---------+-----------------------+---------------------------\n";

    $correct = 0;
    foreach ($testPayloads as $item) {
        $res = classifyWafPayload($network, $item['payload']);
        $expCode = $classInfo[$item['class']]['code'];
        $predCode = $res['class_code'];
        $ok = ($item['class'] === $res['predicted_class']);
        if ($ok) {
            $correct++;
        }

        $badge = $ok ? '✓' : '✗';
        $shortPayload = strlen($item['payload']) > 46
            ? substr($item['payload'], 0, 43) . '...'
            : $item['payload'];

        echo sprintf(
            " %-8s | %s %-7s |  %5.1f%% | %-21s | %s\n",
            $expCode,
            $badge,
            $predCode,
            $res['threat_score'] * 100,
            $res['action'],
            $shortPayload
        );
    }
    echo "------------------------------------------------------------------------------------\n";
    echo sprintf(
        "Unseen Test Suite Accuracy: %d/%d (%.1f%%)\n",
        $correct,
        count($testPayloads),
        ($correct / count($testPayloads)) * 100
    );

    // Show a detailed breakdown on one obfuscated SQLi payload
    printDetailedInspection(
        $network,
        "item_id=42'/**/AND/**/1=0/**/UNION/**/SELECT/**/username,password/**/FROM/**/accounts--"
    );
}

/**
 * Inspects a JSON batch file of payloads.
 */
function inspectBatchFile(Network $network, string $filePath): void
{
    if (!file_exists($filePath)) {
        fwrite(STDERR, "Error: File not found: {$filePath}\n");
        exit(1);
    }

    $items = json_decode((string)file_get_contents($filePath), true);
    if (!is_array($items)) {
        fwrite(STDERR, "Error: Invalid JSON format in {$filePath}\n");
        exit(1);
    }

    echo "\nScanning " . count($items) . " payloads from {$filePath}...\n\n";
    $blocked = 0;
    $allowed = 0;

    foreach ($items as $idx => $entry) {
        $payload = is_array($entry) ? ($entry['payload'] ?? '') : (string)$entry;
        $res = classifyWafPayload($network, $payload);
        if ($res['predicted_class'] === WAF_CLASS_BENIGN) {
            $allowed++;
        } else {
            $blocked++;
        }
        echo sprintf(
            "#%02d [%-6s | %5.1f%%] %-21s %s\n",
            $idx + 1,
            $res['class_code'],
            $res['threat_score'] * 100,
            $res['action'],
            substr($payload, 0, 60)
        );
    }

    echo "\nBatch Summary: {$allowed} Allowed, {$blocked} Flagged/Blocked.\n\n";
}

function printWafHelp(): void
{
    echo <<<HELP

NiNa Neural Web Application Firewall (WAF) Inspector
====================================================

Usage:
  php examples/WAF/inspect_request.php
  php examples/WAF/inspect_request.php "<http_payload>"
  php examples/WAF/inspect_request.php --file <json_file>

Examples:
  php examples/WAF/inspect_request.php
  php examples/WAF/inspect_request.php "id=1' OR '1'='1'--"
  php examples/WAF/inspect_request.php "<svg/onload=alert(document.cookie)>"
  php examples/WAF/inspect_request.php "page=../../../../etc/passwd%00"
  php examples/WAF/inspect_request.php "host=127.0.0.1; cat /etc/passwd"
  php examples/WAF/inspect_request.php --file examples/WAF/sample_payloads.json

HELP;
}
