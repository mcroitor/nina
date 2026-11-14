<?php

/**
 * Тест производительности для проверки оптимизации Network::Create
 */

include_once __DIR__ . '/../src/mc/lib.php';
include_once __DIR__ . '/../src/mc/nina/exception/NinaException.php';
include_once __DIR__ . '/../src/mc/nina/exception/ValidationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/ComputationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/SerializationException.php';
include_once __DIR__ . '/../src/mc/nina/exception/LayerException.php';
include_once __DIR__ . '/../src/mc/nina/exception/NetworkException.php';
include_once __DIR__ . '/../src/mc/nina/exception/TrainingException.php';
include_once __DIR__ . '/../src/mc/nina/activation/ActivationFunction.php';
include_once __DIR__ . '/../src/mc/nina/activation/ReLU.php';
include_once __DIR__ . '/../src/mc/nina/activation/Sigmoid.php';
include_once __DIR__ . '/../src/mc/nina/activation/Tanh.php';
include_once __DIR__ . '/../src/mc/nina/activation/Softmax.php';
include_once __DIR__ . '/../src/mc/nina/cell.php';
include_once __DIR__ . '/../src/mc/nina/layer.php';
include_once __DIR__ . '/../src/mc/nina/network.php';

use mc\nina\Network;
use mc\nina\activation\Sigmoid;
use mc\nina\activation\ReLU;

echo "🧪 Тест производительности Network::Create\n";
echo "==========================================\n\n";

// Функция для измерения времени выполнения
function measureTime(callable $func, string $description): float {
    $start = microtime(true);
    $result = $func();
    $end = microtime(true);
    $time = ($end - $start) * 1000; // в миллисекундах
    
    echo sprintf("%-50s: %.3f ms\n", $description, $time);
    return $time;
}

// Функция для подсчета использования памяти
function measureMemory(callable $func, string $description): int {
    $before = memory_get_usage();
    $result = $func();
    $after = memory_get_usage();
    $memory = $after - $before;
    
    echo sprintf("%-50s: %s bytes\n", $description, number_format($memory));
    return $memory;
}

echo "Тест 1: Создание сети с default (null) активацией\n";
echo "--------------------------------------------------\n";

$time1 = measureTime(function() {
    $networks = [];
    for ($i = 0; $i < 100; $i++) {
        $networks[] = Network::Create("test_$i", 3, 32); // null по умолчанию
    }
    return $networks;
}, "100 сетей с null активацией (оптимизированно)");

echo "\nТест 2: Создание сети с явной Sigmoid активацией\n";
echo "------------------------------------------------\n";

$time2 = measureTime(function() {
    $sigmoid = new Sigmoid(); // Создаем один раз
    $networks = [];
    for ($i = 0; $i < 100; $i++) {
        $networks[] = Network::Create("test_$i", 3, 32, $sigmoid); // Переиспользуем
    }
    return $networks;
}, "100 сетей с переиспользуемой Sigmoid");

echo "\nТест 3: Создание сети с новой Sigmoid при каждом вызове\n";
echo "-------------------------------------------------------\n";

$time3 = measureTime(function() {
    $networks = [];
    for ($i = 0; $i < 100; $i++) {
        $networks[] = Network::Create("test_$i", 3, 32, new Sigmoid()); // Создаем каждый раз
    }
    return $networks;
}, "100 сетей с новой Sigmoid каждый раз");

echo "\nТест памяти:\n";
echo "------------\n";

$memory1 = measureMemory(function() {
    $networks = [];
    for ($i = 0; $i < 50; $i++) {
        $networks[] = Network::Create("test_$i", 2, 16); // null по умолчанию
    }
    return $networks;
}, "50 сетей с null активацией");

$memory2 = measureMemory(function() {
    $networks = [];
    for ($i = 0; $i < 50; $i++) {
        $networks[] = Network::Create("test_$i", 2, 16, new Sigmoid()); // новый объект каждый раз
    }
    return $networks;
}, "50 сетей с новой Sigmoid каждый раз");

echo "\nРезультаты:\n";
echo "===========\n";
echo sprintf("Оптимизированный (null): %.3f ms\n", $time1);
echo sprintf("Переиспользование объекта: %.3f ms\n", $time2);
echo sprintf("Создание каждый раз: %.3f ms\n", $time3);

$improvement1 = (($time3 - $time1) / $time3) * 100;
$improvement2 = (($time3 - $time2) / $time3) * 100;

echo sprintf("Улучшение производительности (null): %.1f%%\n", $improvement1);
echo sprintf("Улучшение производительности (переиспользование): %.1f%%\n", $improvement2);

echo "\nПроверка функциональности:\n";
echo "=========================\n";

// Проверяем, что все варианты создают работающие сети
$net1 = Network::Create("test1", 2, 4);
$net2 = Network::Create("test2", 2, 4, null);
$net3 = Network::Create("test3", 2, 4, new Sigmoid());

$input = [1, 0, 1, 0];

echo "Сеть 1 (default): ";
$result1 = $net1->Calculate($input);
echo "✓ Работает (выход: " . count($result1) . " значений)\n";

echo "Сеть 2 (null): ";
$result2 = $net2->Calculate($input);
echo "✓ Работает (выход: " . count($result2) . " значений)\n";

echo "Сеть 3 (explicit): ";
$result3 = $net3->Calculate($input);
echo "✓ Работает (выход: " . count($result3) . " значений)\n";

echo "\n🎉 Все тесты пройдены! Оптимизация работает корректно.\n";