# NiNa - Библиотека нейронных сетей для PHP

![PHP Версия](https://img.shields.io/badge/PHP-8.0%2B-blue.svg)
![Лицензия](https://img.shields.io/badge/License-MIT-green.svg)
![Тесты](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

**NiNa** — это мощная, но простая PHP библиотека нейронных сетей, предназначенная как для начинающих, так и для продвинутых пользователей. Она предоставляет чистый, объектно-ориентированный API для создания, обучения и использования нейронных сетей с поддержкой множественных активационных функций и комплексной обработкой ошибок.

## ✨ Возможности

- 🧠 **Множественные активационные функции**: Sigmoid, Tanh, ReLU, Softmax
- 🛡️ **Надежная обработка ошибок**: Комплексная система исключений с детальным контекстом ошибок
- 🧪 **Хорошо протестировано**: Обширный набор тестов со 100% успешностью
- 🔧 **Простота в использовании**: Чистый, интуитивный API
- 📊 **Генетический алгоритм**: Встроенный эволюционный подход к обучению
- 💾 **Сериализация**: Сохранение и загрузка обученных моделей
- 🎯 **Гибкая архитектура**: Настраиваемые структуры сетей

## 📦 Установка

### Ручная установка

Клонируйте репозиторий и подключите библиотеку:

```php
<?php
require_once 'src/mc/lib.php';
require_once 'src/mc/nina/cell.php';
require_once 'src/mc/nina/layer.php';
require_once 'src/mc/nina/network.php';
require_once 'src/mc/nina/trainer.php';
require_once 'src/mc/nina/predictor.php';

// Подключить активационные функции
require_once 'src/mc/nina/activation/ActivationFunction.php';
require_once 'src/mc/nina/activation/ReLU.php';
require_once 'src/mc/nina/activation/Sigmoid.php';
// ... другие активации
```

## 🚀 Быстрый старт

### Создание базовой нейронной сети

```php
<?php
use mc\nina\Network;
use mc\nina\Trainer;
use mc\nina\Predictor;

// Создать простую 3-слойную сеть с 5 нейронами в каждом
$network = Network::Create("МояСеть", 3, 5);

echo "Создана сеть: " . $network->GetName() . PHP_EOL;
echo "Слоев: " . $network->GetNrLayers() . PHP_EOL;
echo "Размер слоя: " . $network->GetLayerSize() . PHP_EOL;
```

### Обучение генетическим алгоритмом

```php
<?php
// Создать тренера с популяцией из 50 сетей
$trainer = new Trainer($network, 50);

// Обучающие данные
$input = "привет";
$expected = "мир";

// Обучить сеть
$trainer->Train($input, $expected);

// Получить лучшую сеть
$bestNetwork = $trainer->getBestNetwork();
```

### Создание предсказаний

```php
<?php
// Создать предсказатель с обученной сетью
$predictor = new Predictor($bestNetwork);

// Сделать предсказание
$input = "привет";
$prediction = $predictor->Predict($input);

echo "Вход: {$input}" . PHP_EOL;
echo "Предсказание: {$prediction}" . PHP_EOL;
```

## 🧠 Активационные функции

NiNa поддерживает множественные активационные функции, которые могут применяться к отдельным ячейкам:

### Доступные активационные функции

```php
<?php
use mc\nina\activation\{ReLU, Sigmoid, Tanh, Softmax};
use mc\nina\Cell;

// Создать ячейки с разными активационными функциями
$reluCell = new Cell("relu_ячейка", [0.5, -0.2, 0.8], new ReLU());
$sigmoidCell = new Cell("sigmoid_ячейка", [0.3, 0.7, -0.1], new Sigmoid());
$tanhCell = new Cell("tanh_ячейка", [0.1, -0.4, 0.6], new Tanh());
$softmaxCell = new Cell("softmax_ячейка", [0.2, 0.5, 0.3], new Softmax());

// Вычислить выходы
$input = [1.0, 0.5, -0.3];
echo "ReLU выход: " . $reluCell->Calculate($input) . PHP_EOL;
echo "Sigmoid выход: " . $sigmoidCell->Calculate($input) . PHP_EOL;
echo "Tanh выход: " . $tanhCell->Calculate($input) . PHP_EOL;
echo "Softmax выход: " . $softmaxCell->Calculate($input) . PHP_EOL;
```

### Свойства активационных функций

| Функция | Диапазон | Производная | Применение |
|---------|----------|-------------|------------|
| **ReLU** | [0, ∞) | 1 если x > 0, иначе 0 | Скрытые слои, быстрое обучение |
| **Sigmoid** | (0, 1) | σ(x)(1-σ(x)) | Бинарная классификация, выходные слои |
| **Tanh** | (-1, 1) | 1-tanh²(x) | Скрытые слои, центрированный выход |
| **Softmax** | (0, 1) | Сложная | Многоклассовая классификация |

## 🏗️ Продвинутое использование

### Пользовательская архитектура сети

```php
<?php
use mc\nina\{Cell, Layer, Network};
use mc\nina\activation\{ReLU, Sigmoid};

// Создать пользовательские ячейки с конкретными весами и активациями
$inputCells = [
    new Cell("вход_1", [0.5, 0.3, -0.2], new ReLU()),
    new Cell("вход_2", [0.1, 0.8, 0.4], new ReLU()),
    new Cell("вход_3", [-0.3, 0.6, 0.2], new ReLU())
];

$hiddenCells = [
    new Cell("скрытый_1", [0.7, -0.1, 0.4], new Sigmoid()),
    new Cell("скрытый_2", [0.2, 0.5, -0.6], new Sigmoid())
];

$outputCells = [
    new Cell("выход_1", [0.3, -0.4], new Sigmoid())
];

// Создать слои
$inputLayer = new Layer("ВходнойСлой", $inputCells);
$hiddenLayer = new Layer("СкрытыйСлой", $hiddenCells);
$outputLayer = new Layer("ВыходнойСлой", $outputCells);

// Создать сеть из слоев
$customNetwork = new Network("ПользовательскаяСеть", [$inputLayer, $hiddenLayer, $outputLayer]);
```

### Обработка ошибок

NiNa предоставляет комплексную обработку ошибок с конкретными типами исключений:

```php
<?php
use mc\nina\exception\{
    ValidationException,
    NetworkException,
    LayerException,
    TrainingException,
    SerializationException,
    ComputationException
};

try {
    // Попытка создать сеть с невалидными параметрами
    $network = Network::Create("", 0, -5);
} catch (ValidationException $e) {
    echo "Ошибка валидации: " . $e->getMessage() . PHP_EOL;
    echo "Код ошибки: " . $e->getCode() . PHP_EOL;
    echo "Контекст: " . json_encode($e->getContext()) . PHP_EOL;
} catch (NetworkException $e) {
    echo "Ошибка архитектуры сети: " . $e->getMessage() . PHP_EOL;
}

try {
    // Попытка вычисления с невалидными данными
    $cell = new Cell("тест", [1.0, 2.0]);
    $result = $cell->Calculate([INF, NAN]); // Вызовет ComputationException
} catch (ComputationException $e) {
    echo "Ошибка вычисления: " . $e->getDetailedMessage() . PHP_EOL;
}
```

### Сериализация модели

```php
<?php
// Сохранить обученную сеть
$networkJson = $bestNetwork->Json();
file_put_contents('обученная_модель.json', $networkJson);

// Загрузить сеть из файла
$loadedJson = file_get_contents('обученная_модель.json');
$loadedNetwork = Network::Load($loadedJson);

// Проверить загруженную сеть
echo "Слоев в оригинальной сети: " . $bestNetwork->GetNrLayers() . PHP_EOL;
echo "Слоев в загруженной сети: " . $loadedNetwork->GetNrLayers() . PHP_EOL;
```

## 🧪 Тестирование и валидация

### Запуск тестов

```bash
# Запустить все тесты
php tests/test_cell.php
php tests/test_layer.php
php tests/test_exceptions.php

# Пример вывода:
# =============== ТЕСТЫ ================
# всего: 15, пройдено: 15, провалено: 0
# ============== АССЕРТЫ ===============
# всего: 97, пройдено: 97, провалено: 0
```

### Пример цикла обучения

```php
<?php
function trainXORNetwork() {
    $network = Network::Create("XOR_Сеть", 3, 4);
    $trainer = new Trainer($network, 100);
    
    $trainingData = [
        ['00', '0'],
        ['01', '1'],
        ['10', '1'],
        ['11', '0']
    ];
    
    echo "Обучение XOR сети..." . PHP_EOL;
    
    for ($epoch = 0; $epoch < 1000; $epoch++) {
        foreach ($trainingData as $data) {
            $trainer->Train($data[0], $data[1]);
        }
        
        if ($epoch % 100 == 0) {
            echo "Эпоха {$epoch} завершена" . PHP_EOL;
        }
    }
    
    $bestNetwork = $trainer->getBestNetwork();
    $predictor = new Predictor($bestNetwork);
    
    echo "Тестирование обученной сети:" . PHP_EOL;
    foreach ($trainingData as $data) {
        $prediction = $predictor->Predict($data[0]);
        echo "Вход: {$data[0]} -> Ожидаемое: {$data[1]}, Предсказанное: {$prediction}" . PHP_EOL;
    }
    
    return $bestNetwork;
}

// Запустить обучение
$xorNetwork = trainXORNetwork();
```

## 🎯 Случаи использования

### 1. Распознавание паттернов

```php
<?php
// Обучить сеть распознаванию паттернов в строках
$patterns = [
    ['паттерн_а', 'тип_1'],
    ['паттерн_б', 'тип_2'],
    ['паттерн_в', 'тип_1']
];

$network = Network::Create("РаспознавательПаттернов", 2, 10);
$trainer = new Trainer($network, 50);

foreach ($patterns as $pattern) {
    $trainer->Train($pattern[0], $pattern[1]);
}
```

### 2. Аппроксимация функций

```php
<?php
// Аппроксимировать математические функции
function trainFunctionApproximator() {
    $network = Network::Create("АппроксиматорФункций", 3, 8);
    $trainer = new Trainer($network, 75);
    
    // Генерировать обучающие данные для f(x) = x^2
    for ($i = 0; $i < 100; $i++) {
        $x = mt_rand(-100, 100) / 10.0; // Случайное число от -10 до 10
        $y = $x * $x; // f(x) = x^2
        
        $trainer->Train((string)$x, (string)$y);
    }
    
    return $trainer->getBestNetwork();
}
```

### 3. Классификация данных

```php
<?php
// Простой бинарный классификатор
function createBinaryClassifier() {
    $network = Network::Create("БинарныйКлассификатор", 2, 6);
    $trainer = new Trainer($network, 40);
    
    // Обучающие данные: классифицировать как положительные (1) или отрицательные (0)
    $data = [
        ['положительный_пример_1', '1'],
        ['положительный_пример_2', '1'],
        ['отрицательный_пример_1', '0'],
        ['отрицательный_пример_2', '0']
    ];
    
    foreach ($data as $sample) {
        $trainer->Train($sample[0], $sample[1]);
    }
    
    return $trainer->getBestNetwork();
}
```

## 🛠️ Разработка и вклад

### Структура проекта

```
nina/
├── src/mc/
│   ├── lib.php                    # Утилиты
│   └── nina/
│       ├── cell.php               # Реализация нейронной ячейки
│       ├── layer.php              # Реализация слоя
│       ├── network.php            # Реализация сети
│       ├── trainer.php            # Генетический алгоритм обучения
│       ├── predictor.php          # Интерфейс предсказаний
│       ├── activation/            # Активационные функции
│       │   ├── ActivationFunction.php
│       │   ├── ReLU.php
│       │   ├── Sigmoid.php
│       │   ├── Tanh.php
│       │   └── Softmax.php
│       └── exception/             # Система исключений
│           ├── NinaException.php
│           ├── ValidationException.php
│           ├── NetworkException.php
│           ├── LayerException.php
│           ├── TrainingException.php
│           ├── SerializationException.php
│           └── ComputationException.php
├── tests/                         # Наборы тестов
├── docs/                          # Документация
└── README.md
```

### Система исключений

NiNa использует комплексную систему исключений для лучшей обработки ошибок:

- **ValidationException** (6001-6099): Ошибки валидации входных данных
- **NetworkException** (1001-1099): Проблемы архитектуры сети
- **LayerException** (2001-2099): Проблемы конфигурации слоев
- **TrainingException** (3001-3099): Ошибки процесса обучения
- **SerializationException** (4001-4099): Операции сохранения/загрузки
- **ComputationException** (5001-5099): Ошибки математических вычислений

Для подробной документации по исключениям см. [docs/exceptions.md](docs/exceptions.md).

## 🔗 Справочник API

### Основные классы

- **`Network`**: Основной контейнер нейронной сети
- **`Layer`**: Коллекция нейронных ячеек
- **`Cell`**: Отдельный нейрон с весами и активацией
- **`Trainer`**: Реализация обучения генетическим алгоритмом
- **`Predictor`**: Интерфейс для создания предсказаний

### Ключевые методы

```php
// Создание и управление сетью
Network::Create(string $name, int $layers, int $layerSize): Network
$network->Calculate(array $input): array
$network->Json(): string
Network::Load(string $json): Network

// Обучение и эволюция
new Trainer(Network $network, int $populationSize)
$trainer->Train(string $input, string $expected): void
$trainer->getBestNetwork(): Network

// Предсказание
new Predictor(Network $network)
$predictor->Predict(string $input): string

// Операции с ячейками
new Cell(string $name, array $weights, ActivationFunction $activation)
$cell->Calculate(array $input): float
$cell->Born(int $mutationPercent): Cell
```

## 📚 Дополнительные ресурсы

- **[Документация по исключениям](docs/exceptions.md)**: Комплексное руководство по обработке ошибок
- **[Набор тестов](tests/)**: Примеры и тестовые случаи
- **[Активационные функции](src/mc/nina/activation/)**: Математические реализации

## 🤝 Вклад в проект

Вклады приветствуются! Пожалуйста, не стесняйтесь отправлять issues, запросы функций или pull requests.

### Дорожная карта разработки

- ✅ **v0.2.0**: Активационные функции и система исключений (ЗАВЕРШЕНО)
- 🔄 **v0.3.0**: Backpropagation и функции потерь (В ПРОЦЕССЕ)
- 📋 **v0.4.0**: Продвинутые архитектуры и метрики
- 🎯 **v1.0.0**: Готовый к продакшену релиз

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## 🙏 Благодарности

- Вдохновлен потребностью в простых, образовательных нейронных сетях на PHP
- Построен с фокусом на ясность и простоту понимания
- Разработан как для обучения, так и для практических применений

---

**Счастливых нейронных сетей! 🧠✨**