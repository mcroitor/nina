<?php

/**
 * Унифицированное распознавание буквы "а" - Генетический алгоритм
 * 
 * Этот пример демонстрирует обучение нейронной сети с помощью генетического
 * алгоритма для распознавания буквы "а" на спрайтах 8x8 пикселей.
 * Все параметры загружаются из config.json.
 * 
 * @author NiNa Neural Network Library
 * @version 2.0 - Unified Interface
 * @date 2025-10-13
 */

require_once __DIR__ . '/common_functions.php';

use mc\nina\Network;
use mc\nina\Trainer;

try {
    $startTime = microtime(true);
    
    // Загрузка конфигурации
    $config = loadConfig();
    $genetic = $config['genetic'];
    $model = $config['model'];
    
    printUnifiedHeader(
        "Генетический алгоритм", 
        "Эволюционный подход к обучению нейронной сети с популяционным поиском оптимальных весов"
    );
    
    // Подготовка данных
    $trainingData = prepareTrainingDataFromConfig($config);
    printConfigInfo($config, $trainingData);
    
    // Создание и обучение сети
    $mutationRate = normalizeMutationRate($genetic['mutation_rate']);
    echo "🧬 Генетическое обучение:\n";
    echo "========================\n";
    echo "🔬 Размер популяции: {$genetic['population_size']}\n";
    echo "🔄 Максимум поколений: {$genetic['generations']}\n";  
    echo "🧪 Уровень мутации: {$mutationRate}%\n";
    echo "🎯 Целевая приспособленность: " . round($genetic['target_fitness'] * 100, 1) . "%\n\n";
    
    // Создание модели
    $network = createOrLoadNetwork($config, "letter_a_ga_model.json");
    $initialAccuracy = calculateAccuracy($network, $trainingData, $model['output']['threshold']);
    
    // Создание тренера
    $trainer = new Trainer($network, $genetic['population_size'], $mutationRate);
    
    echo "🚀 Начинаем эволюционное обучение...\n";
    
    $bestFitness = $initialAccuracy;
    $bestOverallNetwork = $network;
    for ($iteration = 1; $iteration <= $genetic['generations']; $iteration++) {
        // Пакетное эволюционное обучение поколения на всех примерах датасета
        $trainer->TrainBatch($trainingData);
        
        // Вычисляем приспособленность как точность
        $currentNetwork = $trainer->getBestNetwork();
        $fitness = calculateAccuracy($currentNetwork, $trainingData, $model['output']['threshold']);
        
        if ($fitness >= $bestFitness) {
            $bestFitness = $fitness;
            $bestOverallNetwork = $currentNetwork;
        }
        
        // Показываем прогресс каждые 5 итераций
        if ($iteration % 5 === 0 || $iteration === 1) {
            echo "Поколение $iteration: лучшая приспособленность = " . round($bestFitness * 100, 1) . "%";
            if ($bestFitness >= $genetic['target_fitness']) {
                echo " 🎯";
            }
            echo "\n";
        }
        
        // Ранняя остановка при достижении цели
        if (isset($genetic['early_stop']) && $genetic['early_stop'] && $bestFitness >= $genetic['target_fitness']) {
            echo "🏆 Достигнута целевая приспособленность на поколении $iteration!\n";
            break;
        }
    }
    
    $endTime = microtime(true);
    $trainingTime = $endTime - $startTime;
    $usedGenerations = min($iteration, $genetic['generations']);
    
    // Получаем лучшую сеть
    $bestNetwork = $bestOverallNetwork;
    $finalAccuracy = calculateAccuracy($bestNetwork, $trainingData, $model['output']['threshold']);
    
    // Результаты обучения
    $trainingResults = [
        'generations' => $usedGenerations,
        'final_accuracy' => $finalAccuracy,
        'final_fitness' => $bestFitness,
        'improvement' => max(0.0, $finalAccuracy - $initialAccuracy),
        'training_time' => $trainingTime
    ];
    
    printTrainingResults("Генетический алгоритм", $trainingResults);
    
    // Тестирование
    $testResults = testNetworkUnified($bestNetwork, $config);
    
    // Сохранение модели
    saveModel($bestNetwork, "letter_a_ga_model.json", $config);
    
    // Итоговая сводка
    printFinalSummary("Генетический алгоритм", $trainingResults, $testResults, $config);
    
    // Дополнительная статистика
    echo "📊 Статистика эволюции:\n";
    echo "=======================\n";
    echo "🥇 Лучшая приспособленность: " . round($bestFitness * 100, 1) . "%\n";
    echo "📈 Поколений использовано: $usedGenerations из {$genetic['generations']}\n";
    echo "🎯 Цель достигнута: " . ($bestFitness >= $genetic['target_fitness'] ? "ДА" : "НЕТ") . "\n";
    echo "⚡ Эффективность: " . round(($usedGenerations / $genetic['generations']) * 100, 1) . "% от максимума\n";

} catch (\Throwable $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Класс: " . get_class($e) . "\n";
    echo "Файл: " . $e->getFile() . ":" . $e->getLine() . "\n";
}

