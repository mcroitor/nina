# 🛡️ NiNa Neural Web Application Firewall (WAF)

A multi-class HTTP request & payload threat classifier built with the **NiNa Neural Network Library** using the **`24 → 32 → 16 → 5` baseline architecture** (`1,360` trainable weights).

---

## 🏗️ Network Architecture (`24 → 32 → 16 → 5`)

| Layer | Neurons | Inputs / Neuron | Weights | Activation | Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Input** | `24` | — | — | — | Normalized `[0, 1]` statistical, structural & lexical signals + bias |
| **Hidden 1** | `32` | `24` | `768` | `Tanh` | Non-linear feature conjunction detectors |
| **Hidden 2** | `16` | `32` | `512` | `Tanh` | High-level attack representation compression |
| **Output** | `5` | `16` | `80` | `Sigmoid` | Threat class probabilities (`BENIGN`, `SQLI`, `XSS`, `LFI`, `RCE`) |
| **Total** | **`53`** | — | **`1,360`** | — | `~32 KB` serialized JSON, `~0.15 ms` inference in PHP |

---

## 🔍 24 Extracted Features (`extractWafFeatures`)

1. **Statistical & Encoding (`f[0]–f[5]`):** Log-normalized length, Shannon byte entropy, special-character density, URL/hex/HTML-entity encoding ratio, digit ratio, uppercase ratio.
2. **Structural Delimiters (`f[6]–f[10]`):** Quote density (`'`, `"`), angle brackets (`<`, `>`), SQL/script comment tokens (`--`, `/*`, `#`), directory traversal sequences (`../`, `..\`), shell chaining operators (`;`, `|`, `&&`, `` ` ``, `$(`).
3. **Attack Signatures (`f[11]–f[22]`):**
   * **SQLi (`f[11]–f[13]`):** SQL query verbs, boolean tautologies (`OR 1=1`), database functions & schema probes (`SLEEP`, `INFORMATION_SCHEMA`).
   * **XSS (`f[14]–f[16]`):** Dangerous HTML tags (`<script`, `<svg`, `<iframe`), DOM event handlers (`onerror=`, `onload=`), JS execution sinks (`javascript:`, `alert(`, `document.cookie`).
   * **LFI (`f[17]–f[18]`):** Sensitive Unix files (`/etc/passwd`, `/proc/self`), Windows/PHP stream wrappers (`win.ini`, `php://filter`, `%00`).
   * **RCE (`f[19]–f[21]`):** Shell binaries (`/bin/sh`, `cmd.exe`, `powershell`), recon commands (`whoami`, `uname`, `curl`, `wget`), subshell/redirection syntax (`$IFS`, `2>&1`).
   * **Benign Structure (`f[22]`):** Structural cleanliness score.
4. **Bias (`f[23]`):** Constant `1.0`.

---

## 🚀 Usage

### 1. Train & Benchmark the WAF Model
```bash
php examples/WAF/train_waf.php
```
Trains the network with `Adam`, L2 weight decay (`1e-4`), gradient norm clipping (`5.0`), and `PlateauSchedule`, then prints per-class Precision/Recall/F1 metrics, a $5 \times 5$ confusion matrix, and inference throughput.

### 2. Run the Live WAF Showcase
```bash
php examples/WAF/inspect_request.php
```

### 3. Inspect Custom Payloads from CLI
```bash
# Benign query with quotes and SQL-like words
php examples/WAF/inspect_request.php "comment=Please select an update option for O'Connor"

# Obfuscated SQL Injection
php examples/WAF/inspect_request.php "id=1'/**/UNION/**/SELECT/**/username,password/**/FROM/**/users--"

# Cross-Site Scripting (XSS)
php examples/WAF/inspect_request.php "<svg/onload=alert(document.cookie)>"

# Path Traversal / LFI
php examples/WAF/inspect_request.php "template=php://filter/convert.base64-encode/resource=../../.env"

# Command Injection (RCE)
php examples/WAF/inspect_request.php "host=127.0.0.1; cat /etc/passwd"

# Scan a JSON batch of payloads
php examples/WAF/inspect_request.php --file examples/WAF/sample_payloads.json
```
