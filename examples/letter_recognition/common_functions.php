<?php

/**
 * Общие функции для унифицированного обучения нейронных сетей
 * 
 * @author NiNa Neural Network Library
 * @version 2.0 - Unified Interface
 * @date 2025-10-13
 */

require_once __DIR__ . '/../../src/autoload.php';

/**
 * Загружает конфигурацию из JSON файла
 */
function loadConfig(string $configPath = __DIR__ . '/config.json'): array {
    if (!file_exists($configPath)) {
        throw new Exception("Файл конфигурации не найден: $configPath");
    }
    
    $configContent = file_get_contents($configPath);
    $config = json_decode($configContent, true);
    
    if (json_last_error() !== JSON_ERROR_NONE) {
        throw new Exception("Ошибка парсинга JSON: " . json_last_error_msg());
    }
    
    return $config;
}

/**
 * Создает спрайт буквы "а" 8x8 (основной паттерн)
 */
function createLetterA(): array {
    return [
        0,1,1,1,1,1,0,0,
        1,1,0,0,0,1,1,0,
        1,0,0,0,0,0,1,0,
        1,0,0,0,0,0,1,0,
        1,1,1,1,1,1,1,0,
        1,0,0,0,0,0,1,0,
        1,0,0,0,0,0,1,0,
        0,0,0,0,0,0,0,0
    ];
}

/**
 * Создает альтернативный спрайт буквы "а" 8x8
 */
function createLetterA2(): array {
    return [
        0,0,1,1,1,1,0,0,
        0,1,0,0,0,0,1,0,
        1,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,1,
        1,1,1,1,1,1,1,1,
        1,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,1
    ];
}

/**
 * Создает спрайт буквы "b" 8x8
 */
function createLetterB(): array {
    return [
        1,1,1,1,1,1,0,0,
        1,0,0,0,0,1,1,0,
        1,0,0,0,0,1,1,0,
        1,1,1,1,1,1,0,0,
        1,0,0,0,0,1,1,0,
        1,0,0,0,0,1,1,0,
        1,1,1,1,1,1,0,0,
        0,0,0,0,0,0,0,0
    ];
}

/**
 * Создает спрайт буквы "c" 8x8
 */
function createLetterC(): array {
    return [
        0,1,1,1,1,1,1,0,
        1,1,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,
        1,0,0,0,0,0,0,0,
        1,1,0,0,0,0,0,0,
        0,1,1,1,1,1,1,0,
        0,0,0,0,0,0,0,0
    ];
}

/**
 * Создает случайный спрайт (шум)
 */
function createRandomSprite(int $inputSize): array {
    $sprite = [];
    for ($i = 0; $i < $inputSize; $i++) {
        $sprite[] = (float)mt_rand(0, 1);
    }
    return $sprite;
}

/**
 * Добавляет шум к паттерну
 */
function addNoise(array $pattern, float $noiseLevel = 0.1): array {
    $noisy = [];
    foreach ($pattern as $pixel) {
        $noise = (mt_rand() / mt_getrandmax() - 0.5) * 2 * $noiseLevel;
        $noisy[] = max(0, min(1, $pixel + $noise));
    }
    return $noisy;
}

/**
 * Расширяет входные данные до размера слоя
 */
function adjustInputToLayerSize(array $input, int $targetSize): array {
    if (count($input) >= $targetSize) {
        return array_slice($input, 0, $targetSize);
    }
    return array_pad($input, $targetSize, 0.0);
}

/**
 * Подготавливает обучающий датасет на основе конфигурации
 */
function prepareTrainingDataFromConfig(array $config): array {
    $dataset = $config['dataset'];
    $model = $config['model'];
    
    $letterPatterns = [
        'a' => [createLetterA(), createLetterA2()],
        'b' => [createLetterB()],
        'c' => [createLetterC()]
    ];
    
    $trainingData = [];
    
    // Положительные примеры (буква 'а')
    $positiveSamples = $dataset['positive_samples'];
    $positiveVariations = $dataset['patterns']['positive_variations'];
    
    for ($i = 0; $i < $positiveSamples; $i++) {
        $patternIndex = $i % $positiveVariations;
        $basePattern = $letterPatterns['a'][$patternIndex];
        $noisyPattern = addNoise($basePattern, $dataset['training_noise_level']);
        $adjustedPattern = adjustInputToLayerSize($noisyPattern, $model['architecture']['layer_size']);
        
        $target = array_fill(0, $model['architecture']['layer_size'], $model['output']['target_positive']);
        
        $trainingData[] = [
            'input' => $adjustedPattern,
            'target' => $target,
            'expected' => true,
            'label' => 'a'
        ];
    }
    
    // Отрицательные примеры
    $negativeSamples = $dataset['negative_samples'];
    $negativeLetters = $dataset['patterns']['negative_letters'];
    $includeRandom = $dataset['patterns']['include_random'];
    
    for ($i = 0; $i < $negativeSamples; $i++) {
        if ($i < count($negativeLetters)) {
            // Используем реальные буквы
            $letter = $negativeLetters[$i % count($negativeLetters)];
            $basePattern = $letterPatterns[$letter][0];
            $noisyPattern = addNoise($basePattern, $dataset['training_noise_level']);
            $label = "letter_$letter";
        } else {
            // Используем случайные паттерны если включены
            if ($includeRandom) {
                $noisyPattern = createRandomSprite($model['architecture']['input_size']);
                $label = 'random_pattern';
            } else {
                // Повторяем существующие буквы
                $letter = $negativeLetters[$i % count($negativeLetters)];
                $basePattern = $letterPatterns[$letter][0];
                $noisyPattern = addNoise($basePattern, $dataset['training_noise_level']);
                $label = "letter_$letter";
            }
        }
        
        $adjustedPattern = adjustInputToLayerSize($noisyPattern, $model['architecture']['layer_size']);
        $target = array_fill(0, $model['architecture']['layer_size'], $model['output']['target_negative']);
        
        $trainingData[] = [
            'input' => $adjustedPattern,
            'target' => $target,
            'expected' => false,
            'label' => $label
        ];
    }
    
    return $trainingData;
}

/**
 * Вычисляет точность сети на датасете
 */
function calculateAccuracy($network, array $trainingData, float $threshold): float {
    $correctPredictions = 0;
    $totalSamples = count($trainingData);
    
    foreach ($trainingData as $sample) {
        $output = $network->Calculate($sample['input']);
        $avgOutput = array_sum($output) / count($output);
        
        $predicted = $avgOutput > $threshold;
        if ($predicted === $sample['expected']) {
            $correctPredictions++;
        }
    }
    
    return $correctPredictions / $totalSamples;
}

/**
 * Подготавливает данные для backpropagation обучения
 */
function prepareBackpropData(array $trainingData, int $layerSize): array {
    $backpropData = [];
    
    foreach ($trainingData as $sample) {
        $target = array_fill(0, $layerSize, $sample['target_avg']);
        $backpropData[] = [$sample['input'], $target];
    }
    
    return $backpropData;
}

/**
 * Унифицированный заголовок для всех методов
 */
function printUnifiedHeader(string $methodName, string $description): void {
    $separator = str_repeat("=", 60);
    echo "\n🧠 Распознавание буквы 'а' - $methodName\n";
    echo "$separator\n";
    echo "$description\n\n";
}

/**
 * Информация о конфигурации и датасете
 */
function printConfigInfo(array $config, array $trainingData): void {
    $model = $config['model'];
    $dataset = $config['dataset'];
    
    echo "📋 Конфигурация модели:\n";
    echo "=======================\n";
    echo "🏗️  Архитектура:\n";
    echo "   └ Входной размер: {$model['architecture']['input_size']}\n";
    echo "   └ Количество слоев: {$model['architecture']['layers']}\n";
    echo "   └ Размер слоя: {$model['architecture']['layer_size']}\n";
    echo "   └ Функция активации: {$model['architecture']['activation_function']}\n";
    echo "   └ Порог классификации: {$model['output']['threshold']}\n\n";
    
    echo "📊 Информация о датасете:\n";
    echo "=========================\n";
    echo "📈 Всего примеров: " . count($trainingData) . "\n";
    
    $positives = array_filter($trainingData, fn($sample) => $sample['expected'] === true);
    $negatives = array_filter($trainingData, fn($sample) => $sample['expected'] === false);
    
    echo "  ✅ Положительные (буква 'а'): " . count($positives) . "\n";
    echo "  ❌ Отрицательные примеры: " . count($negatives) . "\n";
    
    // Группируем по типам
    $labelCounts = [];
    foreach ($trainingData as $sample) {
        $label = $sample['label'];
        $labelCounts[$label] = ($labelCounts[$label] ?? 0) + 1;
    }
    
    foreach ($labelCounts as $label => $count) {
        echo "     - $label: $count\n";
    }
    echo "\n";
}

/**
 * Результаты обучения
 */
function printTrainingResults(string $methodName, array $results): void {
    echo "📈 Результаты обучения ($methodName):\n";
    echo "====================================\n";
    
    if (isset($results['iterations'])) {
        echo "🔄 Итераций: {$results['iterations']}\n";
    }
    if (isset($results['epochs'])) {
        echo "📚 Эпох: {$results['epochs']}\n";
    }
    if (isset($results['generations'])) {
        echo "🧬 Поколений: {$results['generations']}\n";
    }
    if (isset($results['final_loss'])) {
        echo "📉 Финальная ошибка: " . round($results['final_loss'], 6) . "\n";
    }
    if (isset($results['final_accuracy'])) {
        echo "🎯 Финальная точность: " . round($results['final_accuracy'] * 100, 1) . "%\n";
    }
    if (isset($results['improvement'])) {
        echo "📊 Улучшение: +" . round($results['improvement'] * 100, 1) . "%\n";
    }
    if (isset($results['training_time'])) {
        echo "⏱️  Время обучения: " . round($results['training_time'], 2) . " сек\n";
    }
    echo "\n";
}

/**
 * Унифицированное тестирование сети
 */
function testNetworkUnified($network, array $config): array {
    echo "🧪 Тестирование обученной сети:\n";
    echo "===============================\n";
    
    $testNoise = $config['dataset']['test_noise_level'];
    $threshold = $config['model']['output']['threshold'];
    $layerSize = $config['model']['architecture']['layer_size'];
    
    $testCases = [
        ['name' => '🅰️  Буква а (основная)', 'pattern' => createLetterA(), 'expected' => true],
        ['name' => '🅰️  Буква а (альтернативная)', 'pattern' => createLetterA2(), 'expected' => true],
        ['name' => '🅰️  Буква а (с шумом)', 'pattern' => addNoise(createLetterA(), $testNoise), 'expected' => true],
        ['name' => '🅱️  Буква b', 'pattern' => createLetterB(), 'expected' => false],
        ['name' => '🅲  Буква c', 'pattern' => createLetterC(), 'expected' => false],
        ['name' => '🎲 Случайный паттерн', 'pattern' => createRandomSprite($config['model']['architecture']['input_size']), 'expected' => false]
    ];
    
    $correctTests = 0;
    $results = [];
    
    foreach ($testCases as $test) {
        $adjustedPattern = adjustInputToLayerSize($test['pattern'], $layerSize);
        $output = $network->Calculate($adjustedPattern);
        $avgOutput = array_sum($output) / count($output);
        
        $predicted = $avgOutput > $threshold;
        $correct = $predicted === $test['expected'];
        
        if ($correct) $correctTests++;
        
        $status = $correct ? '✅' : '❌';
        $confidence = round($avgOutput, 3);
        $expectedStr = $test['expected'] ? 'буква а' : 'НЕ буква а';
        $predictedStr = $predicted ? 'буква а' : 'НЕ буква а';
        
        echo "$status {$test['name']}: $confidence → $predictedStr (ожидалось: $expectedStr)\n";
        
        $results[] = [
            'name' => $test['name'],
            'confidence' => $confidence,
            'predicted' => $predicted,
            'expected' => $test['expected'],
            'correct' => $correct
        ];
    }
    
    $testAccuracy = ($correctTests / count($testCases)) * 100;
    echo "\n📊 Общая точность на тестах: $correctTests/" . count($testCases) . 
         " (" . round($testAccuracy, 1) . "%)\n\n";
    
    return [
        'accuracy' => $testAccuracy / 100,
        'correct_count' => $correctTests,
        'total_count' => count($testCases),
        'details' => $results
    ];
}

/**
 * Итоговая сводка
 */
function printFinalSummary(string $methodName, array $trainingResults, array $testResults, array $config): void {
    echo "📋 ИТОГОВАЯ СВОДКА - $methodName\n";
    echo "=====================================\n";
    
    if (isset($trainingResults['final_accuracy'])) {
        echo "🎓 Точность на обучающих данных: " . round($trainingResults['final_accuracy'] * 100, 1) . "%\n";
    }
    echo "🧪 Точность на тестовых данных: " . round($testResults['accuracy'] * 100, 1) . "%\n";
    
    if (isset($trainingResults['training_time'])) {
        echo "⏱️  Время обучения: " . round($trainingResults['training_time'], 2) . " сек\n";
    }
    
    if (isset($trainingResults['epochs'])) {
        echo "📚 Использовано эпох: {$trainingResults['epochs']}\n";
    }
    if (isset($trainingResults['generations'])) {
        echo "🧬 Использовано поколений: {$trainingResults['generations']}\n";
    }
    
    // Оценка качества
    $testAccuracy = $testResults['accuracy'];
    if ($testAccuracy >= 0.9) {
        echo "🏆 Результат: ОТЛИЧНО (≥90%) - Превосходная производительность!\n";
    } elseif ($testAccuracy >= 0.8) {
        echo "🥈 Результат: ХОРОШО (≥80%) - Хорошая производительность\n";
    } elseif ($testAccuracy >= 0.7) {
        echo "🥉 Результат: УДОВЛЕТВОРИТЕЛЬНО (≥70%) - Приемлемая производительность\n";
    } else {
        echo "❌ Результат: ТРЕБУЕТ УЛУЧШЕНИЯ (<70%) - Необходима настройка параметров\n";
    }
    
    echo "\n🎉 Обучение методом '$methodName' завершено успешно!\n";
}

/**
 * Сохраняет модель в файл
 */
function saveModel($network, string $filename, array $config): void {
    $outputConfig = $config['output'];
    if (!$outputConfig['save_model']) {
        return;
    }
    
    $modelDir = $outputConfig['model_directory'] ?? './models/';
    if (!is_dir($modelDir)) {
        mkdir($modelDir, 0755, true);
    }
    
    $fullPath = $modelDir . $filename;
    file_put_contents($fullPath, $network->Json());
    echo "💾 Модель сохранена: $fullPath\n\n";
}

/**
 * Создает новую сеть или загружает существующую модель
 */
function createOrLoadNetwork(array $config, string $modelFilename): \mc\nina\Network {
    $model = $config['model'];
    $outputConfig = $config['output'];
    
    // Определяем полный путь к модели
    $modelDir = $outputConfig['model_directory'] ?? './models/';
    $fullPath = $modelDir . $modelFilename;
    
    // Пытаемся загрузить существующую модель
    if (file_exists($fullPath) && $outputConfig['load_existing_model']) {
        echo "🔄 Загрузка существующей модели: $fullPath\n";
        try {
            $modelJson = file_get_contents($fullPath);
            $network = \mc\nina\Network::Load($modelJson);
            echo "✅ Модель успешно загружена!\n\n";
            return $network;
        } catch (Exception $e) {
            echo "⚠️  Ошибка загрузки модели: " . $e->getMessage() . "\n";
            echo "🔄 Создаем новую модель...\n\n";
        }
    }
    
    // Создаем новую модель
    echo "🆕 Создание новой модели...\n";
    $network = \mc\nina\Network::Create(
        $model['name'], 
        $model['architecture']['layers'], 
        $model['architecture']['layer_size']
    );
    echo "✅ Новая модель создана!\n\n";
    
    return $network;
}