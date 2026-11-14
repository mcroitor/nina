# NiNa Neural Network Library - Autoloader

## 📦 Автоматическая загрузка классов

Файл `autoload.php` предоставляет удобную автоматическую загрузку всех классов библиотеки NiNa.

### 🚀 Быстрый старт

```php
<?php
// Подключите autoloader одной строкой
require_once __DIR__ . '/autoload.php';

// Теперь все классы NiNa доступны!
$network = Network::Create('my_network', 3, 10);
$trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.01);
```

### 📋 Что включает autoloader

#### Основные классы (с алиасами)

- `Network` → `mc\nina\Network`
- `Cell` → `mc\nina\Cell` 
- `Layer` → `mc\nina\Layer`

#### Функции активации

- `Sigmoid` → `mc\nina\activation\Sigmoid`
- `ReLU` → `mc\nina\activation\ReLU`
- `Tanh` → `mc\nina\activation\Tanh`
- `Softmax` → `mc\nina\activation\Softmax`

#### Алгоритмы обучения

- `BackpropagationTrainer` → `mc\nina\BackpropagationTrainer`

#### Функции потерь

- `MeanSquaredError` → `mc\nina\loss\MeanSquaredError`
- `CrossEntropy` → `mc\nina\loss\CrossEntropy`

#### Все исключения

- Автоматически загружаются при необходимости
- `ValidationException`, `ComputationException`, etc.

### 🔧 Преимущества

**Раньше (без autoloader):**

```php
// Много строк включений
include_once __DIR__ . '/src/mc/lib.php';
include_once __DIR__ . '/src/mc/nina/exception/NinaException.php';
include_once __DIR__ . '/src/mc/nina/exception/ValidationException.php';
include_once __DIR__ . '/src/mc/nina/exception/ComputationException.php';
include_once __DIR__ . '/src/mc/nina/exception/SerializationException.php';
include_once __DIR__ . '/src/mc/nina/exception/LayerException.php';
include_once __DIR__ . '/src/mc/nina/exception/NetworkException.php';
include_once __DIR__ . '/src/mc/nina/activation/ActivationFunction.php';
include_once __DIR__ . '/src/mc/nina/activation/Sigmoid.php';
include_once __DIR__ . '/src/mc/nina/activation/ReLU.php';
include_once __DIR__ . '/src/mc/nina/activation/Tanh.php';
include_once __DIR__ . '/src/mc/nina/cell.php';
include_once __DIR__ . '/src/mc/nina/layer.php';
include_once __DIR__ . '/src/mc/nina/network.php';
include_once __DIR__ . '/src/mc/nina/loss/LossFunction.php';
include_once __DIR__ . '/src/mc/nina/loss/MeanSquaredError.php';
include_once __DIR__ . '/src/mc/nina/BackpropagationTrainer.php';

// Длинные use statements
use mc\nina\Network;
use mc\nina\BackpropagationTrainer;
use mc\nina\loss\MeanSquaredError;
use mc\nina\activation\Sigmoid;
```

**Сейчас (с autoloader):**

```php
// Одна строка!
require_once __DIR__ . '/autoload.php';

// Короткие имена классов
$network = Network::Create('test', 2, 5);
$trainer = new BackpropagationTrainer($network, new MeanSquaredError(), 0.01);
```

### 📁 Структура проекта

```txt
nina/
├── autoload.php          ← Главный autoloader
├── src/mc/               ← Исходный код библиотеки  
│   ├── lib.php          ← Утилиты
│   └── nina/            ← Основные классы
├── examples/            ← Примеры использования
├── tests/               ← Тесты
├── simple_example.php   ← Простой пример
└── test_autoload.php    ← Тест autoloader
```

### 🧪 Тестирование

Запустите тест autoloader:

```bash
php test_autoload.php
```

Попробуйте простой пример:

```bash  
php simple_example.php
```

### 🔍 Отладка

Для отладки autoloader добавьте константу:

```php
define('NINA_DEBUG', true);
require_once __DIR__ . '/autoload.php';
// Выведет информацию о загрузке
```

### 📝 Примечания

- Autoloader загружает классы **по требованию** (lazy loading)
- Совместим с другими autoloader'ами через `spl_autoload_register()`
- Поддерживает полные имена классов (`mc\nina\Network`) и алиасы (`Network`)
- Автоматически подключает `lib.php` с утилитами

### 🎯 Использование в примерах

Все новые примеры должны использовать autoloader:

```php
<?php
/**
 * Пример распознавания букв с NiNa
 */

// Подключение библиотеки
require_once __DIR__ . '/../../autoload.php';

// Создание сети
$network = Network::Create('letter_recognition', 3, 64);

// Обучение
$trainer = new BackpropagationTrainer(
    $network, 
    new MeanSquaredError(), 
    0.01
);

// ... остальная логика
```

Это делает код чище, проще и легче в поддержке! 🚀
