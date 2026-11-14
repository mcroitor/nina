<?php

/**
 * Унифицированное распознавание буквы "а" - Backpropagation
 * 
 * Этот пример демонстрирует обучение нейронной сети с помощью алгоритма
 * backpropagation для распознавания буквы "а" на спрайтах 8x8 пикселей.
 * Все параметры загружаются из config.json.
 * 
 * @author NiNa Neural Network Library
 * @version 2.0 - Unified Interface
 * @date 2025-10-13
 */

require_once __DIR__ . '/common_functions.php';

use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;

try {
    $startTime = microtime(true);
    
    // Загрузка конфигурации
    $config = loadConfig();
    $backprop = $config['backpropagation'];
    $model = $config['model'];
    
    printUnifiedHeader(
        "Backpropagation", 
        "Градиентный метод обучения с прямым и обратным распространением ошибки"
    );
    
    // Создание сети
    $network = createOrLoadNetwork($config, "letter_a_backprop_model.json");
    
    // Подготовка данных
    $trainingData = prepareTrainingDataFromConfig($config);
    printConfigInfo($config, $trainingData);
    $backpropData = prepareBackpropData($trainingData, $model['architecture']['layer_size']);
    
    // Настройка тренера
    echo "🔧 Настройка Backpropagation тренера:\n";
    echo "====================================\n";
    $lossFunction = new MeanSquaredError();
    $trainer = new BackpropagationTrainer(
        $network,
        $lossFunction,
        $backprop['learning_rate'],
        new \mc\nina\optimizer\Adam()
    );
    
    echo "✅ Параметры обучения:\n";
    echo "  📊 Функция потерь: Mean Squared Error\n";
    echo "  🎯 Скорость обучения: {$backprop['learning_rate']}\n";
    echo "  🛑 Целевая ошибка: {$backprop['target_loss']}\n";
    echo "  🔄 Макс. эпох: {$backprop['max_epochs']}\n\n";
    
    // Обучение
    echo "🚀 Начало обучения:\n";
    echo "==================\n";
    
    $initialAccuracy = calculateAccuracy($network, $trainingData, $model['output']['threshold']);
    echo "📈 Начальная точность: " . round($initialAccuracy * 100, 1) . "%\n\n";
    
    // Пакетное обучение
    $batchSize = $backprop['batch_size'] ?? 1;
    $epochLosses = $trainer->trainBatch($backpropData, $backprop['max_epochs'], $backprop['target_loss'], $batchSize);
    
    $trainingTime = microtime(true) - $startTime;
    $totalEpochs = count($epochLosses);
    $finalLoss = end($epochLosses);
    
    // Получаем обученную сеть
    $trainedNetwork = $trainer->getTrainedNetwork();
    $finalAccuracy = calculateAccuracy($trainedNetwork, $trainingData, $model['output']['threshold']);
    
    // Результаты обучения
    $trainingResults = [
        'epochs' => $totalEpochs,
        'final_accuracy' => $finalAccuracy,
        'final_loss' => $finalLoss,
        'improvement' => $finalAccuracy - $initialAccuracy,
        'training_time' => $trainingTime,
        'convergence_achieved' => $finalLoss <= $backprop['target_loss']
    ];
    
    printTrainingResults("Backpropagation", $trainingResults);
    
    // Тестирование
    $testResults = testNetworkUnified($trainedNetwork, $config);
    
    // Сохранение модели
    saveModel($trainedNetwork, "letter_a_backprop_model.json", $config);
    
    // Итоговая сводка
    printFinalSummary("Backpropagation", $trainingResults, $testResults, $config);
    
    // Дополнительная статистика
    echo "📊 Статистика обучения:\n";
    echo "=======================\n";
    echo "🎯 Финальная ошибка: " . round($finalLoss, 6) . "\n";
    echo "📈 Улучшение точности: +" . round(($finalAccuracy - $initialAccuracy) * 100, 1) . "%\n";
    echo "⚡ Эпох использовано: $totalEpochs из {$backprop['max_epochs']}\n";
    echo "🎪 Сходимость: " . ($finalLoss <= $backprop['target_loss'] ? "ДОСТИГНУТА" : "НЕ ДОСТИГНУТА") . "\n";
    
    if (count($epochLosses) > 1) {
        $initialLoss = $epochLosses[0];
        $convergenceRate = ($initialLoss - $finalLoss) / count($epochLosses);
        echo "📉 Скорость сходимости: " . round($convergenceRate * 1000, 4) . " × 10⁻³ за эпоху\n";
    }

} catch (\Throwable $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Класс: " . get_class($e) . "\n";
    echo "Файл: " . $e->getFile() . ":" . $e->getLine() . "\n";
}
