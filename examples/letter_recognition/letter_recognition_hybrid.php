<?php

/**
 * Унифицированное распознавание буквы "а" - Hybrid (GA + Backpropagation)
 * 
 * Этот пример демонстрирует гибридный подход к обучению нейронной сети,
 * комбинирующий генетический алгоритм и backpropagation для распознавания
 * буквы "а" на спрайтах 8x8 пикселей. Все параметры загружаются из config.json.
 * 
 * @author NiNa Neural Network Library
 * @version 2.0 - Unified Interface
 * @date 2025-10-13
 */

require_once __DIR__ . '/common_functions.php';

use mc\nina\Network;
use mc\nina\Trainer;
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;

try {
    $startTime = microtime(true);
    
    // Загрузка конфигурации
    $config = loadConfig();
    $hybrid = $config['hybrid'];
    $model = $config['model'];
    
    printUnifiedHeader(
        "Hybrid (GA + Backpropagation)", 
        "Двухэтапный подход: генетическая эволюция + градиентная оптимизация"
    );
    
    // Создание сети
    $network = createOrLoadNetwork($config, "letter_a_hybrid_model.json");
    
    // Подготовка данных
    $trainingData = prepareTrainingDataFromConfig($config);
    
    // === ЭТАП 1: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ===
    echo "🧬 ЭТАП 1: Генетический алгоритм\n";
    echo "================================\n";
    echo "Поиск перспективной области весового пространства...\n\n";
    
    $gaTrainer = new Trainer($network, $hybrid['ga_population'], $hybrid['ga_mutation_rate']);
    
    echo "✅ Параметры GA этапа:\n";
    echo "  👥 Популяция: {$hybrid['ga_population']} особей\n";
    echo "  🎲 Мутация: " . ($hybrid['ga_mutation_rate'] * 100) . "%\n";
    echo "  🏁 Поколений: {$hybrid['ga_generations']}\n";
    echo "  🎯 Целевая приспособленность: " . ($hybrid['ga_target_fitness'] * 100) . "%\n\n";
    
    // Обучение генетическим алгоритмом
    $gaIteration = 0;
    $bestFitness = 0;
    
    for ($i = 0; $i < $hybrid['ga_generations']; $i++) {
        foreach ($trainingData as $sample) {
            $expectedOutput = array_fill(0, $model['architecture']['layer_size'], 
                $sample['label'] === 'a' ? $model['output']['target_positive'] : $model['output']['target_negative']
            );
            
            $gaTrainer->Train($sample['input'], $expectedOutput);
        }
        
        $currentFitness = calculateAccuracy($gaTrainer->GetBestNetwork(), $trainingData, $model['output']['threshold']);
        
        if ($currentFitness > $bestFitness) {
            $bestFitness = $currentFitness;
        }
        
        if ($i % 10 === 0 || $currentFitness >= $hybrid['ga_target_fitness']) {
            echo "Поколение " . ($i + 1) . ": Лучшая точность = " . round($currentFitness * 100, 1) . "%\n";
        }
        
        if ($currentFitness >= $hybrid['ga_target_fitness']) {
            $gaIteration = $i + 1;
            echo "🎯 Целевая приспособленность достигнута на поколении $gaIteration!\n";
            break;
        }
        
        $gaIteration = $i + 1;
    }
    
    $gaNetwork = $gaTrainer->GetBestNetwork();
    $gaAccuracy = calculateAccuracy($gaNetwork, $trainingData, $model['output']['threshold']);
    $gaTime = microtime(true) - $startTime;
    
    echo "\n✅ GA этап завершен:\n";
    echo "  ⏱️ Время: " . round($gaTime, 1) . " сек\n";
    echo "  🏆 Финальная точность: " . round($gaAccuracy * 100, 1) . "%\n";
    echo "  📊 Поколений использовано: $gaIteration из {$hybrid['ga_generations']}\n\n";
    
    // === ЭТАП 2: BACKPROPAGATION ===
    echo "🎯 ЭТАП 2: Backpropagation\n";
    echo "==========================\n";
    echo "Точная доводка найденного решения...\n\n";
    
    $bpStartTime = microtime(true);
    $lossFunction = new MeanSquaredError();
    $bpTrainer = new BackpropagationTrainer($gaNetwork, $lossFunction, $hybrid['bp_learning_rate']);
    
    echo "✅ Параметры BP этапа:\n";
    echo "  📊 Функция потерь: Mean Squared Error\n";
    echo "  🎯 Скорость обучения: {$hybrid['bp_learning_rate']}\n";
    echo "  🛑 Целевая ошибка: {$hybrid['bp_target_loss']}\n";
    echo "  🔄 Макс. эпох: {$hybrid['bp_max_epochs']}\n\n";
    
    // Обучение backpropagation
    $epochLosses = $bpTrainer->trainBatch($trainingData, $hybrid['bp_max_epochs'], $hybrid['bp_target_loss']);
    
    $bpTime = microtime(true) - $bpStartTime;
    $totalEpochs = count($epochLosses);
    $finalLoss = end($epochLosses);
    
    $finalNetwork = $bpTrainer->getTrainedNetwork();
    $finalAccuracy = calculateAccuracy($finalNetwork, $trainingData, $model['output']['threshold']);
    
    $trainingTime = microtime(true) - $startTime;
    
    echo "✅ BP этап завершен:\n";
    echo "  ⏱️ Время: " . round($bpTime, 1) . " сек\n";
    echo "  🎯 Финальная ошибка: " . round($finalLoss, 6) . "\n";
    echo "  📊 Эпох использовано: $totalEpochs из {$hybrid['bp_max_epochs']}\n";
    echo "  🎪 Сходимость: " . ($finalLoss <= $hybrid['bp_target_loss'] ? "ДОСТИГНУТА" : "НЕ ДОСТИГНУТА") . "\n\n";
    
    // Результаты обучения
    $trainingResults = [
        'ga_generations' => $gaIteration,
        'bp_epochs' => $totalEpochs,
        'ga_accuracy' => $gaAccuracy,
        'final_accuracy' => $finalAccuracy,
        'final_loss' => $finalLoss,
        'improvement' => ($finalAccuracy - $gaAccuracy) * 100,
        'training_time' => $trainingTime,
        'ga_time' => $gaTime,
        'bp_time' => $bpTime
    ];
    
    printTrainingResults("Hybrid", $trainingResults);
    
    // Тестирование
    $testResults = testNetworkUnified($finalNetwork, $config);
    
    // Сохранение модели
    saveModel($finalNetwork, "letter_a_hybrid_model.json", $config);
    
    // Итоговая сводка
    printFinalSummary("Hybrid", $trainingResults, $testResults, $config);
    
    // Дополнительная статистика
    echo "📊 Статистика гибридного обучения:\n";
    echo "==================================\n";
    echo "🧬 GA этап:\n";
    echo "  📈 Начальная → GA точность: 0% → " . round($gaAccuracy * 100, 1) . "%\n";
    echo "  ⚡ Поколений: $gaIteration из {$hybrid['ga_generations']}\n";
    echo "  ⏱️ Время GA: " . round($gaTime, 1) . " сек\n\n";
    
    echo "🎯 BP этап:\n";
    echo "  📈 GA → Финальная точность: " . round($gaAccuracy * 100, 1) . "% → " . round($finalAccuracy * 100, 1) . "%\n";
    echo "  📉 Улучшение от BP: +" . round(($finalAccuracy - $gaAccuracy) * 100, 1) . "%\n";
    echo "  ⚡ Эпох BP: $totalEpochs из {$hybrid['bp_max_epochs']}\n";
    echo "  ⏱️ Время BP: " . round($bpTime, 1) . " сек\n\n";
    
    echo "🏆 Общий результат:\n";
    echo "  🎯 Финальная точность: " . round($finalAccuracy * 100, 1) . "%\n";
    echo "  ⏱️ Общее время: " . round($trainingTime, 1) . " сек\n";
    echo "  💡 Эффективность гибрида: " . 
        ($finalAccuracy > $gaAccuracy ? "УСПЕШНО" : "НЕОБХОДИМА НАСТРОЙКА") . "\n";

} catch (Exception $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Класс: " . get_class($e) . "\n";
    echo "Файл: " . $e->getFile() . ":" . $e->getLine() . "\n";
}