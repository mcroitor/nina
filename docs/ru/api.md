# API Documentation - NiNa Neural Network Library

## 📚 Оглавление

- [Основные классы](#основные-классы)
  - [Cell](#cell---нейрон)
  - [Layer](#layer---слой)
  - [Network](#network---нейронная-сеть)
  - [Trainer](#trainer---тренер)
  - [Predictor](#predictor---предсказатель)
- [Активационные функции](#активационные-функции)
- [Система исключений](#система-исключений)
- [Утилиты](#утилиты)
- [Примеры использования](#примеры-использования)

---

## Основные классы

### Cell - Нейрон

Базовая единица нейронной сети - отдельный нейрон с весами и активационной функцией.

#### Конструкторы

```php
public function __construct(
    string $name, 
    array $dna, 
    ?ActivationFunction $activationFunction = null
)
```

**Параметры:**

- `$name` (string) - Уникальное имя нейрона
- `$dna` (array) - Массив весов (float значения)
- `$activationFunction` (ActivationFunction|null) - Активационная функция (по умолчанию ReLU)

**Исключения:**

- `ValidationException` - При невалидных параметрах
- `ComputationException` - При некорректных числовых значениях

**Пример:**

```php
use mc\nina\Cell;
use mc\nina\activation\Sigmoid;

$cell = new Cell("neuron_1", [0.5, -0.3, 0.8], new Sigmoid());
```

```php
public static function Create(
    string $name, 
    int $size, 
    float $value = 0.0, 
    ?ActivationFunction $activationFunction = null
): Cell
```

**Параметры:**

- `$name` (string) - Имя нейрона
- `$size` (int) - Количество весов (размер входного слоя)
- `$value` (float) - Начальное значение всех весов
- `$activationFunction` (ActivationFunction|null) - Активационная функция

**Возвращает:** `Cell` - Новый объект нейрона

**Исключения:**

- `ValidationException` - При size <= 0 или невалидном имени
- `ComputationException` - При некорректном значении value

**Пример:**

```php
$cell = Cell::Create("neuron_1", 5, 0.1, new ReLU());
```

#### Основные методы

```php
public function Calculate(array $values): float
```

Вычисляет выход нейрона (прямое распространение с активацией).

**Параметры:**

- `$values` (array) - Входные значения для нейрона

**Возвращает:** `float` - Активированный выход нейрона

**Исключения:**

- `ValidationException` - При несоответствии размеров входа и весов
- `ComputationException` - При численных ошибках или недопустимых значениях

**Пример:**

```php
$input = [1.0, 0.5, -0.3];
$output = $cell->Calculate($input);
echo "Output: {$output}"; // Output: 0.73105
```

```php
public function CalculateRaw(array $values): float
```

Вычисляет взвешенную сумму без применения активационной функции.

**Параметры:**

- `$values` (array) - Входные значения

**Возвращает:** `float` - Сырой выход (до активации)

**Пример:**

```php
$rawOutput = $cell->CalculateRaw($input);
echo "Raw output: {$rawOutput}"; // Raw output: 1.1
```

```php
public function Born(int $percent): Cell
```

Создает новый нейрон на основе текущего с мутацией весов.

**Параметры:**

- `$percent` (int) - Процент мутации (0-100)

**Возвращает:** `Cell` - Новый мутированный нейрон

**Пример:**

```php
$mutatedCell = $cell->Born(10); // 10% мутация
```

#### Методы доступа

```php
public function GetName(): string
```

Возвращает имя нейрона.

```php
public function GetDna(): array
```

Возвращает массив весов нейрона.

```php
public function GetSize(): int
```

Возвращает количество весов (размер входа).

```php
public function GetValue(int $index): float
```

Возвращает конкретный вес по индексу.

```php
public function GetActivationFunction(): ActivationFunction
```

Возвращает активационную функцию нейрона.

```php
public function SetActivationFunction(ActivationFunction $activationFunction): void
```

Устанавливает новую активационную функцию.

#### Сериализация

```php
public function Json(): string
```

Сериализует нейрон в JSON формат.

**Возвращает:** `string` - JSON представление нейрона

**Исключения:**

- `SerializationException` - При ошибках JSON кодирования

```php
public static function Load(string $json): Cell
```

Загружает нейрон из JSON.

**Параметры:**

- `$json` (string) - JSON строка

**Возвращает:** `Cell` - Восстановленный нейрон

**Исключения:**

- `SerializationException` - При ошибках парсинга JSON

---

### Layer - Слой

Коллекция нейронов, работающих параллельно.

#### Конструкторы

```php
public function __construct(string $name, array $cells)
```

**Параметры:**

- `$name` (string) - Имя слоя
- `$cells` (array) - Массив объектов Cell

**Исключения:**

- `ValidationException` - При невалидном имени или неправильных типах ячеек
- `LayerException` - При пустом массиве ячеек

```php
public static function Create(int $size): Layer
```

**Параметры:**

- `$size` (int) - Количество нейронов в слое

**Возвращает:** `Layer` - Новый слой

**Исключения:**

- `LayerException` - При size <= 0

#### Основные методы

```php
public function Calculate(array $values): array
```

Обрабатывает входные данные через все нейроны слоя.

**Параметры:**

- `$values` (array) - Входные данные

**Возвращает:** `array` - Массив выходов от всех нейронов

**Исключения:**

- `ValidationException` - При пустых входных данных
- `LayerException` - При ошибках вычислений в нейронах

```php
public function Born(int $percent): Layer
```

Создает новый слой с мутированными нейронами.

**Параметры:**

- `$percent` (int) - Процент мутации

**Возвращает:** `Layer` - Новый мутированный слой

#### Методы доступа

```php
public function GetName(): string
```

Возвращает имя слоя.

```php
public function GetCell(int $index): Cell
```

Возвращает нейрон по индексу.

**Исключения:**

- `ValidationException` - При недопустимом индексе

```php
public function GetSize(): int
```

Возвращает количество нейронов в слое.

#### Сериализация

```php
public function Json(): string
public static function Load(string $json): Layer
```

Аналогично методам Cell с обработкой массива нейронов.

---

### Network - Нейронная сеть

Полная нейронная сеть из последовательно соединенных слоев.

#### Конструкторы

```php
public function __construct(string $name, array $layers)
```

**Параметры:**

- `$name` (string) - Имя сети
- `$layers` (array) - Массив объектов Layer

**Исключения:**

- `ValidationException` - При невалидном имени или типах слоев
- `NetworkException` - При пустом массиве слоев или несовместимых размерах

```php
public static function Create(string $name, int $nrLayers, int $layerSize): Network
```

**Параметры:**

- `$name` (string) - Имя сети
- `$nrLayers` (int) - Количество слоев
- `$layerSize` (int) - Размер каждого слоя

**Возвращает:** `Network` - Новая сеть

**Исключения:**

- `ValidationException` - При невалидном имени
- `NetworkException` - При неположительных значениях параметров

#### Основные методы

```php
public function Calculate(array $values): array
```

Выполняет прямое распространение через всю сеть.

**Параметры:**

- `$values` (array) - Входные данные

**Возвращает:** `array` - Финальные выходы сети

**Особенности:**

- Автоматически выравнивает размер входных данных до размера первого слоя
- Если входных данных меньше - дополняет нулями
- Если больше - обрезает до нужного размера

**Исключения:**

- `ValidationException` - При пустых входных данных
- `NetworkException` - При ошибках вычислений в слоях

```php
public function Born(int $percent): Network
```

Создает новую сеть с мутированными слоями.

**Параметры:**

- `$percent` (int) - Процент мутации

**Возвращает:** `Network` - Новая мутированная сеть

#### Методы доступа

```php
public function GetName(): string
```

Возвращает имя сети.

```php
public function GetNrLayers(): int
```

Возвращает количество слоев.

```php
public function GetLayerSize(): int
```

Возвращает размер слоев (все слои имеют одинаковый размер).

```php
public function GetLayer(int $index): Layer
```

Возвращает слой по индексу.

**Исключения:**

- `ValidationException` - При недопустимом индексе

#### Сериализация

```php
public function Json(): string
public static function Load(string $json): Network
```

Полная сериализация/десериализация структуры сети.

---

### Trainer - Тренер

Реализует генетический алгоритм для обучения нейронных сетей.

#### Конструктор

```php
public function __construct(Network $network, int $pull)
```

**Параметры:**

- `$network` (Network) - Базовая сеть для эволюции
- `$pull` (int) - Размер популяции (1-1000)

**Исключения:**

- `ValidationException` - При неправильном типе сети
- `TrainingException` - При недопустимом размере популяции

#### Основные методы

```php
public function Train(string $input, string $expected): void
```

Выполняет один цикл обучения с заданными данными.

**Параметры:**

- `$input` (string) - Входные данные для обучения
- `$expected` (string) - Ожидаемый результат

**Алгоритм:**

1. Тестирует все сети в популяции
2. Выбирает лучшую по манхэттенскому расстоянию
3. Создает новую популяцию на основе лучшей сети

**Исключения:**

- `TrainingException` - При пустых данных или ошибках вычислений

```php
public function getBestNetwork(): Network
```

Возвращает лучшую сеть из популяции.

```php
public function getNetworks(): array
```

Возвращает всю текущую популяцию сетей.

```php
public function getNetworkByIndex(int $index): Network
```

Возвращает сеть по индексу из популяции.

**Исключения:**

- `ValidationException` - При недопустимом индексе

---

### Predictor - Предсказатель

Интерфейс для получения предсказаний от обученной сети.

#### Конструктор

```php
public function __construct(Network $network)
```

**Параметры:**

- `$network` (Network) - Обученная сеть

**Исключения:**

- `ValidationException` - При неправильном типе сети

#### Основные методы

```php
public function Predict(string $input): string
```

Получает предсказание от сети.

**Параметры:**

- `$input` (string) - Входные данные

**Возвращает:** `string` - Предсказание сети

**Процесс:**

1. Конвертирует строку в массив чисел
2. Прогоняет через сеть
3. Конвертирует результат обратно в строку

**Исключения:**

- `ValidationException` - При пустых входных данных
- `ComputationException` - При ошибках вычислений или некорректных результатах

---

## Активационные функции

### Интерфейс ActivationFunction

```php
interface ActivationFunction
{
    public function activate(float $input): float;
    public function derivative(float $input): float;
    public function getName(): string;
}
```

### Доступные функции

#### ReLU (Rectified Linear Unit)

```php
use mc\nina\activation\ReLU;

$relu = new ReLU();
$output = $relu->activate(-0.5); // 0.0
$output = $relu->activate(0.5);  // 0.5
```

**Характеристики:**

- Диапазон: [0, ∞)
- Производная: 1 если x > 0, иначе 0
- Применение: Скрытые слои, быстрое обучение

#### Sigmoid

```php
use mc\nina\activation\Sigmoid;

$sigmoid = new Sigmoid();
$output = $sigmoid->activate(0.0); // 0.5
$output = $sigmoid->activate(1.0); // 0.73105
```

**Характеристики:**

- Диапазон: (0, 1)
- Производная: σ(x)(1-σ(x))
- Применение: Бинарная классификация, выходные слои

#### Tanh (Гиперболический тангенс)

```php
use mc\nina\activation\Tanh;

$tanh = new Tanh();
$output = $tanh->activate(0.0);  // 0.0
$output = $tanh->activate(1.0);  // 0.76159
```

**Характеристики:**

- Диапазон: (-1, 1)
- Производная: 1-tanh²(x)
- Применение: Скрытые слои, центрированный выход

#### Softmax

```php
use mc\nina\activation\Softmax;

$softmax = new Softmax();
$output = $softmax->activate(1.0); // Зависит от контекста
```

**Характеристики:**

- Диапазон: (0, 1)
- Применение: Многоклассовая классификация
- Особенность: Требует нормализации по всему слою

---

## Система исключений

### Базовый класс NinaException

```php
abstract class NinaException extends Exception
{
    public function getContext(): array;
    public function getSeverity(): string;
    public function getDetailedMessage(): string;
    public function toArray(): array;
}
```

### Типы исключений

#### ValidationException (6001-6099)

```php
ValidationException::invalidValue(string $parameter, $value, string $expectedType, string $reason)
ValidationException::outOfRange(string $parameter, $value, $min, $max)
ValidationException::missingParameter(string $parameter, string $context)
ValidationException::invalidArraySize(string $parameter, int $actual, int $expected)
```

#### NetworkException (1001-1099)

```php
NetworkException::invalidArchitecture(string $reason, array $context)
NetworkException::configurationError(string $parameter, $value, string $reason)
NetworkException::incompatibleOperation(string $operation, string $reason)
```

#### LayerException (2001-2099)

```php
LayerException::invalidSize(int $size, string $reason)
LayerException::dimensionMismatch(int $expected, int $actual, string $context)
LayerException::invalidConnection(string $fromLayer, string $toLayer, string $reason)
LayerException::operationError(string $operation, string $reason)
```

#### TrainingException (3001-3099)

```php
TrainingException::invalidTrainingData(string $reason, array $context)
TrainingException::convergenceError(int $epoch, float $error, string $reason)
TrainingException::invalidParameters(string $parameter, $value, string $reason)
TrainingException::dataSizeMismatch(int $inputSize, int $expectedSize, string $dataType)
TrainingException::trainingTimeout(int $maxEpochs, float $timeElapsed)
```

#### SerializationException (4001-4099)

```php
SerializationException::fileError(string $operation, string $filename, string $reason)
SerializationException::invalidFormat(string $expectedFormat, string $actualFormat, string $context)
SerializationException::versionMismatch(string $modelVersion, string $libraryVersion)
SerializationException::corruptedData(string $dataType, string $reason)
SerializationException::jsonError(string $operation, int $jsonError)
```

#### ComputationException (5001-5099)

```php
ComputationException::domainError(string $operation, $value, string $reason)
ComputationException::divisionByZero(string $context)
ComputationException::matrixError(string $operation, array $dimensions, string $reason)
ComputationException::numericalOverflow(string $operation, $value, string $type)
ComputationException::invalidNumericalValue($value, string $context)
```

---

## Утилиты

### Функции lib.php

```php
// Создание массивов
function array_create(int $size, float $value = 0.0): array

// Конвертация данных
function string2array(string $input): array
function array2string(array $input): string

// Генетические операции
function mutate(float $value, int $percent): float
function uuid(): string

// Математические функции
function manhattan_distance(array $a, array $b): float
```

---

## Примеры использования

### Базовое создание и обучение

```php
<?php
use mc\nina\{Network, Trainer, Predictor};
use mc\nina\activation\ReLU;

// Создание сети
$network = Network::Create("MyNetwork", 3, 5);

// Создание тренера
$trainer = new Trainer($network, 50);

// Обучение
$trainer->Train("input_data", "expected_output");

// Получение лучшей сети
$bestNetwork = $trainer->getBestNetwork();

// Создание предсказателя
$predictor = new Predictor($bestNetwork);

// Предсказание
$result = $predictor->Predict("test_input");
echo "Prediction: {$result}";
```

### Кастомная архитектура

```php
<?php
use mc\nina\{Cell, Layer, Network};
use mc\nina\activation\{ReLU, Sigmoid};

// Создание кастомных нейронов
$cells = [
    new Cell("c1", [0.5, -0.3, 0.8], new ReLU()),
    new Cell("c2", [0.1, 0.7, -0.2], new Sigmoid())
];

// Создание слоя
$layer = new Layer("CustomLayer", $cells);

// Создание сети
$network = new Network("CustomNetwork", [$layer]);
```

### Обработка ошибок

```php
<?php
use mc\nina\exception\{ValidationException, NetworkException, NinaException};

try {
    $network = Network::Create("", -1, 0); // Невалидные параметры
} catch (ValidationException $e) {
    echo "Validation error: " . $e->getMessage();
    echo "Error code: " . $e->getCode();
    print_r($e->getContext());
} catch (NetworkException $e) {
    echo "Network error: " . $e->getDetailedMessage();
} catch (NinaException $e) {
    echo "General library error: " . $e->getMessage();
    
    // Логирование для отладки
    error_log(json_encode($e->toArray()));
}
```

### Сериализация моделей

```php
<?php
// Сохранение модели
$networkJson = $network->Json();
file_put_contents('model.json', $networkJson);

// Загрузка модели
try {
    $savedJson = file_get_contents('model.json');
    $loadedNetwork = Network::Load($savedJson);
} catch (SerializationException $e) {
    echo "Failed to load model: " . $e->getMessage();
}
```

### Продвинутое обучение

```php
<?php
function trainXORNetwork(): Network {
    $network = Network::Create("XOR", 3, 4);
    $trainer = new Trainer($network, 100);
    
    $trainingData = [
        ['00', '0'], ['01', '1'], 
        ['10', '1'], ['11', '0']
    ];
    
    // Множественные эпохи обучения
    for ($epoch = 0; $epoch < 1000; $epoch++) {
        foreach ($trainingData as [$input, $expected]) {
            try {
                $trainer->Train($input, $expected);
            } catch (TrainingException $e) {
                echo "Training failed at epoch {$epoch}: " . $e->getMessage();
                break 2;
            }
        }
    }
    
    return $trainer->getBestNetwork();
}

$xorNetwork = trainXORNetwork();
```

---

## 📊 Коды возврата и статусы

### HTTP-подобные коды ошибок

- **1xxx**: Ошибки сети (NetworkException)
- **2xxx**: Ошибки слоев (LayerException)  
- **3xxx**: Ошибки обучения (TrainingException)
- **4xxx**: Ошибки сериализации (SerializationException)
- **5xxx**: Ошибки вычислений (ComputationException)
- **6xxx**: Ошибки валидации (ValidationException)

### Уровни серьезности

- **error**: Критические ошибки, требующие вмешательства
- **warning**: Предупреждения, работа может продолжаться
- **critical**: Катастрофические ошибки, требующие остановки

---

## 🔗 Дополнительные ресурсы

- [Архитектурная документация](architecture.md)
- [Руководство по исключениям](exceptions.md)
- [Примеры и туториалы](../examples/)
- [Тесты](../tests/)

---

*Документация актуальна для NiNa v0.2.0+*
