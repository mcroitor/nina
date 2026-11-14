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
    echo "🧬 Генетическое обучение:\n";
    echo "========================\n";
    echo "🔬 Размер популяции: {$genetic['population_size']}\n";
    echo "🔄 Максимум поколений: {$genetic['generations']}\n";  
    echo "🧪 Уровень мутации: {$genetic['mutation_rate']}%\n";
    echo "🎯 Целевая приспособленность: " . round($genetic['target_fitness'] * 100, 1) . "%\n\n";
    
    // Создание модели
    $network = Network::Create(
        $model['name'], 
        $model['architecture']['layers'], 
        $model['architecture']['layer_size']
    );
    
    // Создание тренера
    $trainer = new Trainer($network, $genetic['population_size'], $genetic['mutation_rate'] * 100);
    
    echo "🚀 Начинаем эволюционное обучение...\n";
    
    $bestFitness = 0;
    for ($iteration = 1; $iteration <= $genetic['generations']; $iteration++) {
        // Обучение на всех примерах в поколении
        foreach ($trainingData as $sample) {
            // Конвертируем массив в строку для NiNa API
            $inputString = implode('', array_map(function($x) { 
                return chr(intval($x * 255)); 
            }, $sample['input']));
            
            $expectedString = implode('', array_map(function($x) { 
                return chr(intval($x * 255)); 
            }, $sample['target']));
            
            $trainer->Train($inputString, $expectedString);
        }
        
        // Вычисляем приспособленность как точность
        $bestNetwork = $trainer->getBestNetwork();
        $fitness = calculateAccuracy($bestNetwork, $trainingData, $model['output']['threshold']);
        
        if ($fitness > $bestFitness) {
            $bestFitness = $fitness;
        }
        
        // Показываем прогресс каждые 5 итераций
        if ($iteration % 5 === 0 || $iteration === 1) {
            echo "Поколение $iteration: лучшая приспособленность = " . round($fitness * 100, 1) . "%";
            if ($fitness >= $genetic['target_fitness']) {
                echo " 🎯";
            }
            echo "\n";
        }
        
        // Ранняя остановка при достижении цели
        if (isset($genetic['early_stop']) && $genetic['early_stop'] && $fitness >= $genetic['target_fitness']) {
            echo "🏆 Достигнута целевая приспособленность на поколении $iteration!\n";
            break;
        }
    }
    
    $endTime = microtime(true);
    $trainingTime = $endTime - $startTime;
    
    // Получаем лучшую сеть
    $bestNetwork = $trainer->getBestNetwork();
    $finalAccuracy = calculateAccuracy($bestNetwork, $trainingData, $model['output']['threshold']);
    
    // Результаты обучения
    $trainingResults = [
        'generations' => $iteration,
        'final_accuracy' => $finalAccuracy,
        'final_fitness' => $bestFitness,
        'improvement' => $bestFitness - 0, // Начальная приспособленность обычно близка к 0
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
    echo "📈 Поколений использовано: $iteration из {$genetic['generations']}\n";
    echo "🎯 Цель достигнута: " . ($bestFitness >= $genetic['target_fitness'] ? "ДА" : "НЕТ") . "\n";
    echo "⚡ Эффективность: " . round(($iteration / $genetic['generations']) * 100, 1) . "% от максимума\n";

} catch (Exception $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Класс: " . get_class($e) . "\n";
    echo "Файл: " . $e->getFile() . ":" . $e->getLine() . "\n";
}

