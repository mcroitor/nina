<?php

/**
 * Тестирование обученной нейронной сети распознавания букв
 * 
 * Этот скрипт загружает сохраненную модель нейронной сети и тестирует её
 * на заданном паттерне. Поддерживает различные форматы входных данных.
 * 
 * Использование:
 * php test_letter_recognition.php <network_file> <pattern_file>
 * 
 * Параметры:
 * - network_file: путь к JSON файлу с сохраненной моделью
 * - pattern_file: путь к файлу с паттерном для тестирования
 * 
 * Формат файла паттерна:
 * - JSON массив чисел: [0,1,1,0,1,0,0,1,...]
 * - Текстовый формат: построчно, где 0/1 представляют пиксели
 * 
 * @author Mihail Croitor
 * @version 1.0
 * @date 2025-10-10
 */

// Подключение автозагрузчика библиотеки
require_once __DIR__ . '/../../src/autoload.php';

use mc\nina\Network;
use mc\nina\Predictor;

/**
 * Расширяет входной массив до нужного размера добавлением нулей
 */
function expandInputToLayerSize(array $input, int $targetSize): array
{
    if (count($input) >= $targetSize) {
        return array_slice($input, 0, $targetSize);
    }
    
    return array_pad($input, $targetSize, 0);
}

/**
 * Загружает паттерн из файла
 * Поддерживает JSON и текстовый формат
 */
function loadPattern(string $patternFile): array
{
    if (!file_exists($patternFile)) {
        throw new Exception("Файл паттерна не найден: $patternFile");
    }
    
    $content = file_get_contents($patternFile);
    if ($content === false) {
        throw new Exception("Не удалось прочитать файл паттерна: $patternFile");
    }
    
    $content = trim($content);
    
    // Пробуем загрузить как JSON
    $jsonData = json_decode($content, true);
    if ($jsonData !== null && is_array($jsonData)) {
        echo "✓ Паттерн загружен из JSON формата\n";
        return array_map('floatval', $jsonData);
    }
    
    // Пробуем загрузить как текстовый формат (построчно)
    $lines = explode("\n", $content);
    $pattern = [];
    
    foreach ($lines as $line) {
        $line = trim($line);
        if (empty($line)) continue;
        
        // Разбиваем строку на символы
        for ($i = 0; $i < strlen($line); $i++) {
            $char = $line[$i];
            if ($char === '0' || $char === '1') {
                $pattern[] = (int)$char;
            } elseif (ctype_space($char)) {
                // Пропускаем пробелы
                continue;
            }
        }
    }
    
    if (empty($pattern)) {
        throw new Exception("Не удалось распознать формат файла паттерна. Поддерживаются JSON массив или текстовый формат с 0/1.");
    }
    
    echo "✓ Паттерн загружен из текстового формата\n";
    return $pattern;
}

/**
 * Отображает паттерн как спрайт 8x8 (если размер позволяет)
 */
function displayPattern(array $pattern, string $title = ""): void
{
    if ($title) {
        echo "$title:\n";
    }
    
    // Берем первые 64 элемента для отображения как 8x8
    $displayPattern = array_slice($pattern, 0, 64);
    $displayPattern = array_pad($displayPattern, 64, 0);
    
    for ($row = 0; $row < 8; $row++) {
        for ($col = 0; $col < 8; $col++) {
            echo $displayPattern[$row * 8 + $col];
        }
        echo "\n";
    }
    echo "\n";
}

/**
 * Анализирует результат сети для классификации
 */
function analyzeResult(array $result): array
{
    $average = array_sum($result) / count($result);
    $isLetterA = $average > 0.5;
    
    // Дополнительная статистика
    $min = min($result);
    $max = max($result);
    $positiveCount = count(array_filter($result, fn($x) => $x > 0.5));
    $clampedAvg = min(1.0, max(0.0, $average));
    $confidence = $isLetterA ? $clampedAvg : (1.0 - $clampedAvg);
    
    return [
        'average' => $average,
        'min' => $min,
        'max' => $max,
        'positive_outputs' => $positiveCount,
        'total_outputs' => count($result),
        'is_letter_a' => $isLetterA,
        'confidence' => $confidence
    ];
}

// Проверка аргументов командной строки
if ($argc < 3) {
    echo "Использование: php test_letter_recognition.php <network_file> <pattern_file>\n\n";
    echo "Параметры:\n";
    echo "  network_file  - путь к JSON файлу с сохраненной моделью\n";
    echo "  pattern_file  - путь к файлу с паттерном для тестирования\n\n";
    echo "Пример:\n";
    echo "  php test_letter_recognition.php letter_a_recognition_model.json pattern_a.txt\n\n";
    echo "Формат файла паттерна:\n";
    echo "  JSON: [0,1,1,0,1,0,0,1,...]\n";
    echo "  Текст: построчно с символами 0 и 1\n";
    exit(1);
}

$networkFile = $argv[1];
$patternFile = $argv[2];

echo "🧪 Тестирование нейронной сети распознавания букв\n";
echo "================================================\n\n";

try {
    // Шаг 1: Загрузка нейронной сети
    echo "Шаг 1: Загрузка нейронной сети...\n";
    
    if (!file_exists($networkFile)) {
        throw new Exception("Файл сети не найден: $networkFile");
    }
    
    $modelJson = file_get_contents($networkFile);
    if ($modelJson === false) {
        throw new Exception("Не удалось прочитать файл сети: $networkFile");
    }
    
    $network = Network::Load($modelJson);
    
    echo "✓ Сеть успешно загружена:\n";
    echo "  - Название: " . $network->GetName() . "\n";
    echo "  - Количество слоев: " . $network->GetNrLayers() . "\n";
    echo "  - Размер слоя: " . $network->GetLayerSize() . "\n\n";
    
    // Шаг 2: Загрузка паттерна
    echo "Шаг 2: Загрузка паттерна для тестирования...\n";
    
    $pattern = loadPattern($patternFile);
    $originalSize = count($pattern);
    
    echo "✓ Паттерн загружен:\n";
    echo "  - Исходный размер: $originalSize элементов\n";
    
    // Отображаем паттерн как спрайт
    displayPattern($pattern, "Загруженный паттерн (8x8)");
    
    // Расширяем паттерн до размера слоя сети
    $layerSize = $network->GetLayerSize();
    $expandedPattern = expandInputToLayerSize($pattern, $layerSize);
    
    echo "  - Расширен до размера сети: " . count($expandedPattern) . " элементов\n\n";
    
    // Шаг 3: Тестирование сети
    echo "Шаг 3: Тестирование паттерна через сеть...\n";
    
    $result = $network->Calculate($expandedPattern);
    $analysis = analyzeResult($result);
    
    echo "✓ Результат обработки:\n";
    echo sprintf("  - Среднее значение выходов: %.6f\n", $analysis['average']);
    echo sprintf("  - Минимальное значение: %.6f\n", $analysis['min']);
    echo sprintf("  - Максимальное значение: %.6f\n", $analysis['max']);
    echo sprintf("  - Активных выходов (>0.5): %d из %d\n", 
                 $analysis['positive_outputs'], 
                 $analysis['total_outputs']);
    echo sprintf("  - Уверенность классификации: %.2f%%\n", $analysis['confidence'] * 100);
    
    // Шаг 4: Итоговая классификация
    echo "\nШаг 4: Итоговая классификация...\n";
    
    if ($analysis['is_letter_a']) {
        echo "🎯 РЕЗУЛЬТАТ: Распознана как буква 'А'\n";
        echo "   Сеть считает, что входной паттерн соответствует букве 'а'\n";
    } else {
        echo "❌ РЕЗУЛЬТАТ: НЕ является буквой 'А'\n";
        echo "   Сеть считает, что входной паттерн НЕ соответствует букве 'а'\n";
    }
    
    echo sprintf("   Уверенность: %.2f%%\n", $analysis['confidence'] * 100);
    
    // Шаг 5: Тестирование через Predictor
    echo "\nШаг 5: Дополнительное тестирование через Predictor...\n";
    
    $predictor = new Predictor($network);
    
    // Конвертируем паттерн в строку для Predictor
    $patternString = "";
    foreach ($expandedPattern as $pixel) {
        $patternString .= chr(min(255, max(0, (int)round($pixel * 255))));
    }
    
    $predictionString = $predictor->Predict($patternString);
    $predValues = \mc\string2array($predictionString);
    $avgPred = array_sum($predValues) / count($predValues);
    
    echo sprintf("✓ Predictor результат: %.6f\n", $avgPred);
    echo sprintf("  - Классификация: %s\n", 
                 $avgPred > 0.5 ? "Буква 'А'" : "НЕ буква 'А'");
    echo sprintf("  - Уверенность Predictor: %.2f%%\n", 
                 ($avgPred > 0.5 ? $avgPred : (1 - $avgPred)) * 100);
    
    echo "\n🎉 Тестирование завершено успешно!\n";
    
} catch (\Throwable $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Стек вызовов:\n" . $e->getTraceAsString() . "\n";
    exit(1);
}