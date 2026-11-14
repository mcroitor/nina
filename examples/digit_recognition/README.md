# 🔢 Multi-Class Digit Recognition (0–9)

This example demonstrates **10-class digit classification (`0–9`)** on `5×7` pixel bitmaps using a **heterogeneous neural network** built with the NiNa library.

---

## ✨ Key Features Demonstrated

* **Heterogeneous Network Architecture:**
  * **Input:** `35` pixels (`5×7` bitmap grid)
  * **Hidden Layer:** `24` neurons with `Tanh` activation (`XavierInitializer`)
  * **Output Layer:** `10` neurons with `Sigmoid` activation (one-hot encoded classes `0–9`)
* **Adaptive Optimization:** Trained with `BackpropagationTrainer`, `Adam` optimizer, and `StepDecaySchedule`.
* **Multi-Font & Noise Robustness:** Trained on canonical bitmaps, alternative LCD/7-segment font variants, and noisy samples.
* **Visual CLI Inspector:** Displays `5×7` ASCII sprites (`██`, `▒▒`, `··`) and a 10-class activation bar chart.

---

## 🚀 Quick Start

### 1. Train the Model
```bash
php examples/digit_recognition/train_digits.php
```
This trains the network in ~1 second, saves `examples/digit_recognition/digit_model.json`, and generates sample pattern files `digit_0.txt` through `digit_9.txt` in `examples/digit_recognition/patterns/`.

### 2. Run the Full 0–9 Showcase
```bash
php examples/digit_recognition/recognize_digit.php
```

### 3. Classify a Pattern File or Noisy Digit
```bash
# Recognize from a 5x7 text pattern file
php examples/digit_recognition/recognize_digit.php examples/digit_recognition/patterns/digit_8.txt

# Recognize a specific digit with synthetic pixel noise (0.0 - 0.5)
php examples/digit_recognition/recognize_digit.php --digit 7 --noise 0.2
```

---

## 📁 Directory Structure

* `common.php` — `5×7` digit bitmaps (canonical & LCD variants), network builder, dataset generator, and evaluation helpers.
* `train_digits.php` — Training script with `Adam` and `StepDecaySchedule`.
* `recognize_digit.php` — CLI inference tool and visual showcase.
* `digit_model.json` — Pre-trained heterogeneous neural network model.
* `patterns/digit_0.txt` .. `digit_9.txt` — Editable `5×7` text bitmaps for testing.
