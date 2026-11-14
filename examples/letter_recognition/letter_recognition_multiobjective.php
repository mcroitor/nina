<?php

/**
 * Унифицированное распознавание буквы "а" - Multiobjective Optimization
 * 
 * Этот пример демонстрирует многокритериальную оптимизацию для обучения
 * нейронной сети распознавания буквы "а" на спрайтах 8x8 пикселей.
 * Все параметры загружаются из config.json.
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
    $multiobjective = $config['multiobjective'];
    $model = $config['model'];
    
    printUnifiedHeader(
        "Multiobjective Optimization", 
        "Многокритериальная оптимизация с балансировкой точности и устойчивости"
    );
    
    // Создание сети
    $network = createOrLoadNetwork($config, "letter_a_multiobjective_model.json");
    
    // Подготовка данных
    $trainingData = prepareTrainingDataFromConfig($config);
    printConfigInfo($config, $trainingData);
    
    // Инициализация многокритериальной популяции
    echo "🎯 Настройка многокритериальной оптимизации:\n";
    echo "============================================\n";
    
    $populations = [];
    $bestNetworks = [];
    $objectives = [];
    $mutationRate = normalizeMutationRate($multiobjective['mutation_rate']);
    
    // Создаем отдельные популяции для разных целей
    $objectiveNames = ['accuracy', 'robustness', 'simplicity'];
    
    foreach ($objectiveNames as $objective) {
        $populations[$objective] = new Trainer(
            createOrLoadNetwork($config, "temp_$objective.json"), 
            $multiobjective['population_per_objective'], 
            $mutationRate
        );
        $objectives[$objective] = [
            'fitness' => 0,
            'weight' => $multiobjective['objective_weights'][$objective]
        ];
    }
    
    echo "✅ Параметры многокритериальной оптимизации:\n";
    echo "  👥 Популяция на цель: {$multiobjective['population_per_objective']} особей\n";
    echo "  🎲 Мутация: {$mutationRate}%\n";
    echo "  🏁 Поколений: {$multiobjective['generations']}\n";
    echo "  ⚖️ Веса целей:\n";
    foreach ($multiobjective['objective_weights'] as $obj => $weight) {
        echo "    • " . ucfirst($obj) . ": " . ($weight * 100) . "%\n";
    }
    echo "\n";
    
    // Многокритериальная эволюция
    echo "🚀 Начало многокритериальной эволюции:\n";
    echo "======================================\n";
    
    $generationResults = [];
    
    for ($generation = 0; $generation < $multiobjective['generations']; $generation++) {
        $generationStart = microtime(true);
        
        // Обучение каждой популяции с фокусом на свою цель
        foreach ($objectiveNames as $objective) {
            $trainer = $populations[$objective];
            $trainer->TrainBatch($trainingData);
            
            // Вычисление фитнеса для данной цели
            $currentNetwork = $trainer->getBestNetwork();
            
            switch ($objective) {
                case 'accuracy':
                    $fitness = calculateAccuracy($currentNetwork, $trainingData, $model['output']['threshold']);
                    break;
                    
                case 'robustness':
                    // Тестируем устойчивость к шуму
                    $noisyData = addNoiseToDataset($trainingData, 0.1);
                    $fitness = calculateAccuracy($currentNetwork, $noisyData, $model['output']['threshold']);
                    break;
                    
                case 'simplicity':
                    // Простота = обратная сложность (меньше активных связей = лучше)
                    $accuracy = calculateAccuracy($currentNetwork, $trainingData, $model['output']['threshold']);
                    $complexity = calculateNetworkComplexity($currentNetwork);
                    $fitness = $accuracy * (1 - $complexity); // Баланс точности и простоты
                    break;
            }
            
            $objectives[$objective]['fitness'] = $fitness;
            $bestNetworks[$objective] = $currentNetwork;
        }
        
        // Вычисляем комбинированную приспособленность
        $combinedFitness = 0;
        foreach ($objectiveNames as $objective) {
            $weight = $objectives[$objective]['weight'];
            $fitness = $objectives[$objective]['fitness'];
            $combinedFitness += $weight * $fitness;
        }
        
        // Выбор лучшей сети среди всех целей
        $bestOverallFitness = -1;
        $bestOverallNetwork = null;
        
        foreach ($objectiveNames as $objective) {
            if ($objectives[$objective]['fitness'] > $bestOverallFitness) {
                $bestOverallFitness = $objectives[$objective]['fitness'];
                $bestOverallNetwork = $bestNetworks[$objective];
            }
        }
        
        $generationTime = microtime(true) - $generationStart;
        
        $generationResults[$generation] = [
            'combined_fitness' => $combinedFitness,
            'objectives' => $objectives,
            'best_network' => $bestOverallNetwork,
            'time' => $generationTime
        ];
        
        if ($generation % 10 === 0 || $generation === $multiobjective['generations'] - 1) {
            echo "Поколение " . ($generation + 1) . ":\n";
            echo "  🎯 Комбинированная приспособленность: " . round($combinedFitness * 100, 1) . "%\n";
            foreach ($objectiveNames as $objective) {
                echo "  📊 " . ucfirst($objective) . ": " . 
                     round($objectives[$objective]['fitness'] * 100, 1) . "%\n";
            }
            echo "  ⏱️ Время: " . round($generationTime, 2) . " сек\n\n";
        }
        
        // Проверка на раннюю остановку
        if ($combinedFitness >= $multiobjective['target_combined_fitness']) {
            echo "🎯 Целевая комбинированная приспособленность достигнута на поколении " . ($generation + 1) . "!\n\n";
            break;
        }
    }
    
    $evolutionTime = microtime(true) - $startTime;
    
    // Финальная оптимизация лучшей сети
    echo "🎯 Финальная оптимизация лучшей сети:\n";
    echo "=====================================\n";
    
    $finalOptimizationStart = microtime(true);
    
    $finalResults = end($generationResults);
    $bestFinalNetwork = $finalResults['best_network'];
    $finalCombinedFitness = $finalResults['combined_fitness'];
    
    // Краткий BP для полировки лучшего решения
    if ($multiobjective['final_bp_epochs'] > 0) {
        echo "Применяем финальную BP оптимизацию ({$multiobjective['final_bp_epochs']} эпох)...\n";
        
        $lossFunction = new MeanSquaredError();
        $finalTrainer = new BackpropagationTrainer(
            $bestFinalNetwork, 
            $lossFunction, 
            $multiobjective['final_bp_learning_rate'],
            new \mc\nina\optimizer\Adam()
        );
        
        $backpropData = prepareBackpropData($trainingData, $model['architecture']['layer_size']);
        $epochLosses = $finalTrainer->trainBatch(
            $backpropData, 
            $multiobjective['final_bp_epochs'], 
            0.01
        );
        
        $bestFinalNetwork = $finalTrainer->getTrainedNetwork();
        $finalLoss = end($epochLosses);
        echo "✅ BP оптимизация завершена, финальная ошибка: " . round($finalLoss, 6) . "\n\n";
    }
    
    $finalOptimizationTime = microtime(true) - $finalOptimizationStart;
    $totalTrainingTime = microtime(true) - $startTime;
    
    // Финальная оценка
    $finalAccuracy = calculateAccuracy($bestFinalNetwork, $trainingData, $model['output']['threshold']);
    $finalRobustness = calculateAccuracy($bestFinalNetwork, addNoiseToDataset($trainingData, 0.1), $model['output']['threshold']);
    $finalComplexity = calculateNetworkComplexity($bestFinalNetwork);
    $finalCombinedFitness = 
        $multiobjective['objective_weights']['accuracy'] * $finalAccuracy +
        $multiobjective['objective_weights']['robustness'] * $finalRobustness +
        $multiobjective['objective_weights']['simplicity'] * ($finalAccuracy * (1 - $finalComplexity));
    
    // Результаты обучения
    $trainingResults = [
        'generations' => count($generationResults),
        'epochs' => $multiobjective['final_bp_epochs'] ?? 0,
        'final_accuracy' => $finalAccuracy,
        'final_robustness' => $finalRobustness,
        'final_complexity' => $finalComplexity,
        'combined_fitness' => $finalCombinedFitness,
        'training_time' => $totalTrainingTime,
        'evolution_time' => $evolutionTime,
        'final_optimization_time' => $finalOptimizationTime
    ];
    
    printTrainingResults("Multiobjective", $trainingResults);
    
    // Тестирование
    $testResults = testNetworkUnified($bestFinalNetwork, $config);
    
    // Сохранение модели
    saveModel($bestFinalNetwork, "letter_a_multiobjective_model.json", $config);
    
    // Итоговая сводка
    printFinalSummary("Multiobjective", $trainingResults, $testResults, $config);
    
    // Дополнительная статистика
    echo "📊 Статистика многокритериальной оптимизации:\n";
    echo "=============================================\n";
    echo "🎯 Финальные метрики:\n";
    echo "  📈 Точность: " . round($finalAccuracy * 100, 1) . "%\n";
    echo "  🛡️ Устойчивость: " . round($finalRobustness * 100, 1) . "%\n";
    echo "  🎲 Простота: " . round((1 - $finalComplexity) * 100, 1) . "%\n";
    echo "  ⚖️ Комбинированная приспособленность: " . round($finalCombinedFitness * 100, 1) . "%\n\n";
    
    echo "⏱️ Временные метрики:\n";
    echo "  🧬 Эволюция: " . round($evolutionTime, 1) . " сек\n";
    echo "  🎯 Финальная оптимизация: " . round($finalOptimizationTime, 1) . " сек\n";
    echo "  🏁 Общее время: " . round($totalTrainingTime, 1) . " сек\n\n";
    
    echo "📈 Эффективность подхода:\n";
    echo "  🎪 Поколений использовано: " . count($generationResults) . " из {$multiobjective['generations']}\n";
    echo "  💡 Баланс целей: " . 
        (($finalAccuracy > 0.8 && $finalRobustness > 0.7) ? "ДОСТИГНУТ" : "ТРЕБУЕТ НАСТРОЙКИ") . "\n";

} catch (\Throwable $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Класс: " . get_class($e) . "\n";
    echo "Файл: " . $e->getFile() . ":" . $e->getLine() . "\n";
}

// Вспомогательные функции для многокритериальной оптимизации

function addNoiseToDataset(array $dataset, float $noiseLevel): array
{
    $noisyDataset = [];
    foreach ($dataset as $sample) {
        $noisyInput = [];
        foreach ($sample['input'] as $value) {
            $noise = (mt_rand() / mt_getrandmax() - 0.5) * 2 * $noiseLevel;
            $noisyInput[] = max(0.0, min(1.0, $value + $noise));
        }
        $noisyDataset[] = [
            'input' => $noisyInput,
            'target' => $sample['target'],
            'expected' => $sample['expected'],
            'label' => $sample['label']
        ];
    }
    return $noisyDataset;
}

function calculateNetworkComplexity(Network $network): float
{
    // Оцениваем среднюю абсолютную величину весов сети в диапазоне [0, 1]
    $layers = $network->GetNrLayers();
    $totalWeightMagnitude = 0.0;
    $totalWeights = 0;

    for ($l = 0; $l < $layers; $l++) {
        $layer = $network->GetLayer($l);
        for ($c = 0; $c < $layer->GetSize(); $c++) {
            $dna = $layer->GetCell($c)->GetDna();
            foreach ($dna as $w) {
                $totalWeightMagnitude += abs($w);
                $totalWeights++;
            }
        }
    }

    if ($totalWeights === 0) {
        return 0.0;
    }

    return min(1.0, $totalWeightMagnitude / $totalWeights);
}