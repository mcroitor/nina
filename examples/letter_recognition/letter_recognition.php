<?php

/**
 * Распознавание буквы "а" на спрайте 8x8 пикселей
 * 
 * Этот пример демонстрирует создание нейронной сети для классификации
 * изображений букв размером 8x8 пикселей. Модель состоит из 3 слоев
 * по 100 нейронов каждый и использует генетический алгоритм для обучения.
 * 
 * Особенности архитектуры:
 * - Вход: 64 значения (пиксели спрайта 8x8), расширенные до 100 элементов
 * - 3 скрытых слоя по 100 нейронов
 * - Выход: 100 значений (среднее определяет класс)
 * - Функция активации: ReLU (скрытые слои), Sigmoid (выходной слой)
 * - Метод обучения: Генетический алгоритм
 * 
 * @author NiNa Neural Network Library
 * @version 1.0
 * @date 2025-10-10
 */

// Подключение всех необходимых файлов
include_once __DIR__ . '/../../src/mc/lib.php';
include_once __DIR__ . '/../../src/mc/nina/exception/NinaException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/ValidationException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/ComputationException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/SerializationException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/LayerException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/NetworkException.php';
include_once __DIR__ . '/../../src/mc/nina/exception/TrainingException.php';
include_once __DIR__ . '/../../src/mc/nina/activation/ActivationFunction.php';
include_once __DIR__ . '/../../src/mc/nina/activation/ReLU.php';
include_once __DIR__ . '/../../src/mc/nina/activation/Sigmoid.php';
include_once __DIR__ . '/../../src/mc/nina/activation/Tanh.php';
include_once __DIR__ . '/../../src/mc/nina/activation/Softmax.php';
include_once __DIR__ . '/../../src/mc/nina/cell.php';
include_once __DIR__ . '/../../src/mc/nina/layer.php';
include_once __DIR__ . '/../../src/mc/nina/network.php';
include_once __DIR__ . '/../../src/mc/nina/trainer.php';
include_once __DIR__ . '/../../src/mc/nina/predictor.php';

use mc\nina\Network;
use mc\nina\Trainer;
use mc\nina\Predictor;

echo "🎯 Распознавание буквы 'а' - Нейронная сеть 8x8\n";
echo "================================================\n\n";

// Конфигурация модели
$NR_LAYERS = 3;             // Количество слоев в сети
$LAYER_SIZE = 100;           // Размер каждого слоя в сети
$model_name = 'a_recognition';
$model_file = "{$model_name}_model.json";
$TRAINING_ITERATIONS = 1000;  // Количество итераций обучения
$TRAINER_POPULATION = 200; // Размер популяции для генетического алгоритма

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
 * Создает спрайт буквы "а" 8x8 (1 = пиксель, 0 = фон)
 */
function createLetterA(): array
{
    return [
        0,1,1,1,1,1,0,0,  // Верх буквы "а"
        1,1,0,0,0,1,1,0,  // Боковые стороны
        1,0,0,0,0,0,1,0,  // Внутренняя часть
        1,0,0,0,0,0,1,0,  // Внутренняя часть
        1,1,1,1,1,1,1,0,  // Горизонтальная линия
        1,0,0,0,0,0,1,0,  // Ноги буквы
        1,0,0,0,0,0,1,0,  // Ноги буквы
        0,0,0,0,0,0,0,0   // Пустая строка
    ];
}

/**
 * Создает спрайт буквы "а" 8x8 (1 = пиксель, 0 = фон)
 * Вариант 2 с небольшими изменениями
 */
function createLetterA2(): array
{
    return [
        0,0,1,1,1,1,0,0,  // Верх буквы "а"
        0,1,0,0,0,0,1,0,  // Боковые стороны
        1,0,0,0,0,0,0,1,  // Внутренняя часть
        1,0,0,0,0,0,0,1,  // Внутренняя часть
        1,1,1,1,1,1,1,1,  // Горизонтальная линия
        1,0,0,0,0,0,0,1,  // Ноги буквы
        1,0,0,0,0,0,0,1,  // Ноги буквы
        1,0,0,0,0,0,0,1   // Пустая строка
    ];
}

/**
 * Создает случайный спрайт (не буква "а")
 */
function createRandomSprite(): array
{
    $sprite = [];
    for ($i = 0; $i < 64; $i++) {
        $sprite[] = mt_rand(0, 1);
    }
    return $sprite;
}

/**
 * Создает спрайт другой буквы (например, "b")
 */
function createLetterB(): array
{
    return [
        1,1,1,1,1,1,0,0,  // Верх буквы "b"
        1,0,0,0,0,1,1,0,  // Боковые стороны
        1,0,0,0,0,1,1,0,  // Внутренняя часть
        1,1,1,1,1,1,0,0,  // Средняя линия
        1,0,0,0,0,1,1,0,  // Нижняя часть
        1,0,0,0,0,1,1,0,  // Нижняя часть
        1,1,1,1,1,1,0,0,  // Низ буквы
        0,0,0,0,0,0,0,0   // Пустая строка
    ];
}
function createLetterB2(): array
{
    return [
        1,1,1,1,1,1,1,0,  // Верх буквы "b"
        1,1,0,0,0,0,1,1,  // Боковые стороны
        1,1,0,0,0,0,1,1,  // Внутренняя часть
        1,1,1,1,1,1,0,0,  // Средняя линия
        1,1,0,0,0,0,1,1,  // Нижняя часть
        1,1,0,0,0,0,1,1,  // Нижняя часть
        1,1,0,0,0,0,1,1,  // Нижняя часть
        1,1,1,1,1,1,1,0,  // Низ буквы
    ];
}

/**
 * Отображает спрайт 8x8 в консоли
 */
function displaySprite(array $sprite, string $title = ""): void
{
    if ($title) {
        echo "$title:\n";
    }
    for ($row = 0; $row < 8; $row++) {
        for ($col = 0; $col < 8; $col++) {
            echo $sprite[$row * 8 + $col];
        }
        echo "\n";
    }
    echo "\n";
}

try {
    // Шаг 1: Загрузка или создание нейронной сети
    $modelPath = __DIR__ . '/' . $model_file;

    if (file_exists($modelPath)) {
        echo "Шаг 1: Загрузка существующей модели...\n";
        echo "Найден файл модели: $model_file\n";

        $modelJson = file_get_contents($modelPath);
        $network = Network::Load($modelJson);

        echo "✓ Модель успешно загружена:\n";
        echo "  - Название: " . $network->GetName() . "\n";
        echo "  - Количество слоев: " . $network->GetNrLayers() . "\n";
        echo "  - Размер слоя: " . $network->GetLayerSize() . "\n";
        echo "  - Модель будет дообучена\n\n";
    } else {
        echo "Шаг 1: Создание новой нейронной сети...\n";
        echo "Файл модели не найден, создаем новую сеть\n";
        echo "Архитектура: 64 входа (8x8) -> 3 слоя по $LAYER_SIZE нейронов каждый\n\n";

        // Входной размер = 64 (8x8 пикселей), но будет расширен до размера слоя
        $inputSize = 64;

        // Создаем сеть (все слои имеют одинаковый размер)
        $network = Network::Create($model_name, $NR_LAYERS, $LAYER_SIZE);

        echo "✓ Нейронная сеть создана:\n";
    }

    // Шаг 2: Подготовка обучающих данных
    echo "Шаг 2: Подготовка обучающих данных...\n";

    // Создаем примеры букв "а"
    $letterA = createLetterA();
    displaySprite($letterA, "Буква 'а'");

    $letterA2 = createLetterA2();
    displaySprite($letterA2, "Буква 'а' (вариант 2)");

    // Создаем примеры других символов
    $letterB = createLetterB();
    displaySprite($letterB, "Буква 'b' (не 'а')");
    $letterB2 = createLetterB2();
    displaySprite($letterB2, "Буква 'b' (вариант 2)");

    $randomSprite = createRandomSprite();
    displaySprite($randomSprite, "Случайный спрайт (не 'а')");

    // Расширяем входные данные до размера слоя
    $letterA = expandInputToLayerSize($letterA, $LAYER_SIZE);
    $letterA2 = expandInputToLayerSize($letterA2, $LAYER_SIZE);
    $letterB = expandInputToLayerSize($letterB, $LAYER_SIZE);
    $letterB2 = expandInputToLayerSize($letterB2, $LAYER_SIZE);
    $randomSprite = expandInputToLayerSize($randomSprite, $LAYER_SIZE);

    echo "✓ Обучающие данные подготовлены и расширены до размера слоя ($LAYER_SIZE элементов)\n\n";

    // Шаг 3: Создание тренера
    echo "Шаг 3: Создание тренера с генетическим алгоритмом...\n";
    $trainer = new Trainer($network, $TRAINER_POPULATION);
    echo "✓ Тренер создан с популяцией: {$TRAINER_POPULATION} сетей\n\n";

    // Шаг 4: Обучение на разных примерах
    echo "Шаг 4: Обучение сети...\n";

    // Создаем строки из спрайтов для тренера (Trainer работает со строками символов)
    // Конвертируем массивы [0,1,1,0...] в строки символов для NiNa
    $letterAString = "";
    foreach($letterA as $pixel) {
        $letterAString .= chr($pixel * 255); // 0 -> chr(0), 1 -> chr(255)
    }

    $letterA2String = "";
    foreach($letterA2 as $pixel) {
        $letterA2String .= chr($pixel * 255);
    }

    $letterBString = "";
    foreach($letterB as $pixel) {
        $letterBString .= chr($pixel * 255);
    }

    $letterB2String = "";
    foreach($letterB2 as $pixel) {
        $letterB2String .= chr($pixel * 255);
    }
    
    $randomString = "";
    foreach($randomSprite as $pixel) {
        $randomString .= chr($pixel * 255);
    }

    // Создаем ожидаемые результаты как строки символов размера $LAYER_SIZE
    // Для буквы "а" - символы с высоким ASCII (близко к 255)
    $expectedAString = str_repeat(chr(230), $LAYER_SIZE); // ~0.9 * 255 = 230
    $expectedA2String = str_repeat(chr(230), $LAYER_SIZE);

    // Для НЕ буквы "а" - символы с низким ASCII (близко к 0)
    $expectedNotAString = str_repeat(chr(25), $LAYER_SIZE); // ~0.1 * 255 = 25

    // Проводим обучение с заданным количеством итераций
    echo "Начинаем обучение на $TRAINING_ITERATIONS итераций...\n";
    for ($i = 0; $i < $TRAINING_ITERATIONS; $i++) {
        echo "Итерация обучения: " . ($i + 1) . "/$TRAINING_ITERATIONS\n";
        echo "Обучение на букве 'а' (ожидаемый результат: все выходы ~0.9)...\n";
        $trainer->Train($letterAString, $expectedAString);

        echo "Обучение на букве 'а' (вариант 2) (ожидаемый результат: все выходы ~0.9)...\n";
        $trainer->Train($letterA2String, $expectedA2String);

        echo "Обучение на букве 'b' (ожидаемый результат: все выходы ~0.1)...\n";
        $trainer->Train($letterBString, $expectedNotAString);

        echo "Обучение на букве 'b' (вариант 2) (ожидаемый результат: все выходы ~0.1)...\n";
        $trainer->Train($letterB2String, $expectedNotAString);

        echo "Обучение на случайном спрайте (ожидаемый результат: все выходы ~0.1)...\n";
        $trainer->Train($randomString, $expectedNotAString);
    }

    echo "✓ Обучение завершено!\n\n";

    // Получение лучшей сети после обучения
    $bestNetwork = $trainer->getBestNetwork();

    // Шаг 5: Тестирование сети
    echo "Шаг 5: Тестирование сети...\n";

    // Тестируем на букве "а" (берем среднее значение всех выходов)
    $resultA = $bestNetwork->Calculate($letterA);
    $predictionA = array_sum($resultA) / count($resultA); // Среднее значение

    echo sprintf(
        "Тест буквы 'а': %.3f (среднее из %d выходов) %s\n",
        $predictionA,
        count($resultA),
        ($predictionA > 0.5) ? "✓ Распознана как 'а'" : "✗ Не распознана"
    );

    // Тестируем на букве "а" (вариант 2)
    $resultA2 = $bestNetwork->Calculate($letterA2);
    $predictionA2 = array_sum($resultA2) / count($resultA2);

    echo sprintf(
        "Тест буквы 'а' (вариант 2): %.3f (среднее из %d выходов) %s\n",
        $predictionA2,
        count($resultA2),
        ($predictionA2 > 0.5) ? "✓ Распознана как 'а'" : "✗ Не распознана"
    );

    // Тестируем на букве "b"  
    $resultB = $bestNetwork->Calculate($letterB);
    $predictionB = array_sum($resultB) / count($resultB);

    echo sprintf(
        "Тест буквы 'b': %.3f (среднее из %d выходов) %s\n",
        $predictionB,
        count($resultB),
        ($predictionB < 0.5) ? "✓ Правильно отклонена" : "✗ Ложное срабатывание"
    );

    // Тестируем на букве "b" (вариант 2)
    $resultB2 = $bestNetwork->Calculate($letterB2);
    $predictionB2 = array_sum($resultB2) / count($resultB2);

    echo sprintf(
        "Тест буквы 'b' (вариант 2): %.3f (среднее из %d выходов) %s\n",
        $predictionB2,
        count($resultB2),
        ($predictionB < 0.5) ? "✓ Правильно отклонена" : "✗ Ложное срабатывание"
    );

    // Тестируем на случайном спрайте
    $resultRandom = $bestNetwork->Calculate($randomSprite);
    $predictionRandom = array_sum($resultRandom) / count($resultRandom);

    echo sprintf(
        "Тест случайного спрайта: %.3f (среднее из %d выходов) %s\n",
        $predictionRandom,
        count($resultRandom),
        ($predictionRandom < 0.5) ? "✓ Правильно отклонен" : "✗ Ложное срабатывание"
    );

    echo "\n";

    // Шаг 6: Использование Predictor для удобных предсказаний
    echo "Шаг 6: Использование Predictor для удобных предсказаний...\n";
    $predictor = new Predictor($bestNetwork);

    $predStrA = $predictor->Predict($letterAString);
    $predStrA2 = $predictor->Predict($letterA2String);
    $predStrB = $predictor->Predict($letterBString);
    $predStrB2 = $predictor->Predict($letterB2String);
    $predStrRandom = $predictor->Predict($randomString);

    // Анализируем только первый символ результата как вероятность буквы 'а'
    $firstCharA = ord($predStrA[0]) / 255.0; // Нормализуем ASCII в диапазон [0,1]
    $firstCharA2 = ord($predStrA2[0]) / 255.0;
    $firstCharB = ord($predStrB[0]) / 255.0;
    $firstCharB2 = ord($predStrB2[0]) / 255.0;
    $firstCharRandom = ord($predStrRandom[0]) / 255.0;

    echo "Predictor результаты (анализ первого символа):\n";
    echo sprintf(
        "  Буква 'а': %.3f %s\n",
        $firstCharA,
        ($firstCharA > 0.5) ? "✓ Распознана как 'а'" : "✗ Не распознана"
    );

    echo sprintf(
        "  Буква 'а' (вариант 2): %.3f %s\n",
        $firstCharA2,
        ($firstCharA2 > 0.5) ? "✓ Распознана как 'а'" : "✗ Не распознана"
    );

    echo sprintf(
        "  Буква 'b': %.3f %s\n",
        $firstCharB,
        ($firstCharB < 0.5) ? "✓ Правильно отклонена" : "✗ Ложное срабатывание"
    );

    echo sprintf(
        "  Буква 'b' (вариант 2): %.3f %s\n",
        $firstCharB2,
        ($firstCharB2 < 0.5) ? "✓ Правильно отклонена" : "✗ Ложное срабатывание"
    );

    echo sprintf(
        "  Случайный: %.3f %s\n",
        $firstCharRandom,
        ($firstCharRandom < 0.5) ? "✓ Правильно отклонен" : "✗ Ложное срабатывание"
    );

    echo "\nПолные строки результатов (для справки):\n";
    echo "  Буква 'а': " . substr($predStrA, 0, 20) . "...\n";
    echo "  Буква 'а' (вариант 2): " . substr($predStrA2, 0, 20) . "...\n";
    echo "  Буква 'b': " . substr($predStrB, 0, 20) . "...\n";
    echo "  Буква 'b' (вариант 2): " . substr($predStrB2, 0, 20) . "...\n";
    echo "  Случайный: " . substr($predStrRandom, 0, 20) . "...\n\n";

    // Шаг 7: Сохранение модели (только для новых моделей)
    echo "Шаг 7: Сохранение обученной модели...\n";
    $modelJson = $bestNetwork->Json();
    file_put_contents($modelPath, $modelJson);
    echo "✓ Модель сохранена в '$modelPath'\n\n";

    // Информация о сети
    echo "Информация о сети:\n";
    echo "  Название: " . $bestNetwork->GetName() . "\n";
    echo "  Количество слоев: " . $bestNetwork->GetNrLayers() . "\n";
    echo "  Размер слоя: " . $bestNetwork->GetLayerSize() . "\n\n";

    echo "🎉 Модель распознавания буквы 'а' успешно создана и обучена!\n";
    echo "\nПримечание: Для лучших результатов требуется больше обучающих данных\n";
    echo "и более длительное обучение. Это демонстрационный пример архитектуры.\n";
} catch (Exception $e) {
    echo "❌ Ошибка: " . $e->getMessage() . "\n";
    echo "Стек вызовов:\n" . $e->getTraceAsString() . "\n";
}
