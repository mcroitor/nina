# Генетический алгоритм обучения в NiNa

## 📋 Обзор

Библиотека NiNa использует **генетический алгоритм** для обучения нейронных сетей. Это эволюционный подход, который имитирует естественный отбор для оптимизации весов нейронной сети.

## 🧬 Принципы генетического алгоритма

### Основная концепция

Генетический алгоритм работает с **популяцией** нейронных сетей, где каждая сеть представляет "особь" (индивидуум). Алгоритм итеративно улучшает популяцию через процессы, аналогичные эволюции:

1. **Инициализация** - создание начальной популяции
2. **Оценка fitness** - тестирование каждой особи
3. **Селекция** - выбор лучших особей
4. **Размножение** - создание потомков с мутациями
5. **Замещение** - формирование новой популяции

### Преимущества подхода

- ✅ **Не требует градиентов** - подходит для любых функций активации
- ✅ **Глобальная оптимизация** - избегает локальных минимумов
- ✅ **Простота реализации** - понятный и интуитивный алгоритм
- ✅ **Устойчивость к шуму** - работает с зашумленными данными
- ✅ **Параллелизуемость** - каждую особь можно оценивать независимо

## 🏗️ Архитектура реализации

### Класс `Trainer`

Главный класс для генетического обучения. Управляет популяцией сетей и процессом эволюции.

```php
class Trainer {
    private array $networks = [];      // Популяция сетей
    private Network $best_network;     // Лучшая сеть
    
    public function __construct(Network $network, int $population_size);
    public function Train(string $input, string $expected): void;
    public function getBestNetwork(): Network;
}
```

### Вспомогательные компоненты

- **`Network::Born()`** - создание потомков с мутациями
- **`Layer::Born()`** - мутация слоев
- **`mutate()`** - функция мутации весов
- **`manhattan_distance()`** - функция оценки fitness

## 🔄 Подробный алгоритм

### 1. Инициализация популяции

```php
public function __construct(Network $network, int $pull) {
    $this->best_network = $network;
    
    // Создание популяции мутантов базовой сети
    for ($i = 0; $i < $pull; ++$i) {
        $this->networks[] = $network->Born(5); // 5% мутации
    }
}
```

**Процесс:**
1. Принимается базовая сеть как "родитель"
2. Создается популяция указанного размера (обычно 10-100 особей)
3. Каждая особь - это мутированная копия базовой сети
4. Базовая сеть сохраняется как текущая лучшая

### 2. Функция fitness (оценка качества)

```php
private static function Test(Network $network, string $input, string $expected): float {
    $predictor = new Predictor($network);
    $calculated = $predictor->Predict($input);
    
    // Используем Manhattan distance как меру ошибки
    return manhattan_distance($calculated, $expected);
}
```

**Метрика качества:**
- **Manhattan distance** между предсказанием и ожидаемым результатом
- **Меньшее значение = лучшая особь**
- Формула: `∑|predicted[i] - expected[i]|`

### 3. Основной цикл обучения

```php
public function Train(string $input, string $expected): void {
    $best = $this->best_network;
    $bestScore = self::Test($best, $input, $expected);
    
    // Тестирование всей популяции
    foreach ($this->networks as $network) {
        $score = self::Test($network, $input, $expected);
        if ($score < $bestScore) {
            $best = $network;
            $bestScore = $score;
        }
    }
    
    // Обновление лучшей сети
    $this->best_network = $best;
    
    // Создание новой популяции от лучшей сети
    $this->networks = [];
    for ($i = 0; $i < count($this->networks); ++$i) {
        $this->networks[] = $this->best_network->Born(10); // 10% мутации
    }
}
```

**Пошаговый процесс:**

1. **Оценка популяции** - каждая сеть тестируется на тренировочных данных
2. **Селекция лучшей** - выбирается особь с минимальной ошибкой
3. **Элитизм** - лучшая особь становится новым "родителем"
4. **Генерация потомков** - создается новая популяция от лучшей сети
5. **Мутации** - каждый потомок получает случайные изменения весов

## 🧬 Механизм мутации

### Создание потомков (`Network::Born`)

```php
public function Born(int $percent): Network {
    $layers = [];
    foreach ($this->layers as $layer) {
        $layers[] = $layer->Born($percent); // Мутация каждого слоя
    }
    return new Network($layers, uuid());
}
```

### Мутация слоев (`Layer::Born`)

```php
public function Born(int $percent): Layer {
    $cells = [];
    foreach ($this->cells as $cell) {
        $newWeights = [];
        foreach ($cell->getWeights() as $weight) {
            $newWeights[] = mutate($weight, $percent); // Мутация весов
        }
        $cells[] = new Cell($cell->getName() . "_child", $newWeights, $cell->getActivationFunction());
    }
    return new Layer($cells, uuid());
}
```

### Функция мутации весов

```php
function mutate(float $value, int $precision = 0): float {
    $rand = random_int(0, 100);
    if($rand < $precision) {
        $value += 0.001 * rand(-100, 100); // ±0.1 изменение
    }
    return limit($value); // Ограничение [0, 1]
}
```

**Параметры мутации:**
- **`$precision`** - вероятность мутации каждого веса (0-100%)
- **Сила мутации** - случайное изменение ±0.1
- **Ограничения** - веса остаются в диапазоне [0, 1]

## 📊 Параметры алгоритма

### Размер популяции

```php
$trainer = new Trainer($network, $populationSize);
```

**Рекомендации:**
- **Малые задачи:** 10-20 особей
- **Средние задачи:** 50-100 особей  
- **Сложные задачи:** 100-500 особей

**Компромиссы:**
- ✅ **Больше особей** = лучшее исследование пространства решений
- ❌ **Больше особей** = больше вычислений на итерацию

### Уровень мутации

```php
// В конструкторе - низкая мутация для разнообразия
$network->Born(5);  // 5% вероятность мутации

// В Train - более агрессивная мутация
$network->Born(10); // 10% вероятность мутации
```

**Стратегии мутации:**
- **Консервативная (1-5%)** - тонкая настройка весов
- **Умеренная (10-20%)** - баланс исследования/эксплуатации
- **Агрессивная (30-50%)** - широкое исследование пространства

## 🎯 Практические примеры

### Пример 1: Простое обучение

```php
// Создание базовой сети
$network = Network::Create([10, 5, 1]); // 10 входов, 5 скрытых, 1 выход

// Инициализация тренера с популяцией 20 особей
$trainer = new Trainer($network, 20);

// Обучение на одном примере
$trainer->Train("hello", "world");

// Получение лучшей сети
$bestNetwork = $trainer->getBestNetwork();
```

### Пример 2: Пакетное обучение

```php
$trainer = new Trainer($network, 50);

$trainingData = [
    ["input1", "output1"],
    ["input2", "output2"], 
    ["input3", "output3"]
];

// Несколько эпох обучения
for ($epoch = 0; $epoch < 100; $epoch++) {
    foreach ($trainingData as [$input, $expected]) {
        $trainer->Train($input, $expected);
    }
}

$bestNetwork = $trainer->getBestNetwork();
```

### Пример 3: Мониторинг прогресса

```php
$trainer = new Trainer($network, 30);

for ($epoch = 0; $epoch < 50; $epoch++) {
    foreach ($trainingData as [$input, $expected]) {
        $trainer->Train($input, $expected);
    }
    
    // Проверка прогресса
    $bestNetwork = $trainer->getBestNetwork();
    $predictor = new Predictor($bestNetwork);
    $prediction = $predictor->Predict($testInput);
    $error = manhattan_distance($prediction, $testExpected);
    
    echo "Epoch {$epoch}: Error = {$error}\n";
    
    // Критерий остановки
    if ($error < 0.01) {
        echo "Достигнута целевая точность!\n";
        break;
    }
}
```

## ⚡ Оптимизация производительности

### Настройка параметров

```php
// Адаптивная мутация
class AdaptiveTrainer extends Trainer {
    private int $epochsWithoutImprovement = 0;
    private float $lastBestScore = PHP_FLOAT_MAX;
    
    public function Train(string $input, string $expected): void {
        $currentScore = $this->getBestScore();
        
        // Увеличиваем мутацию при застое
        $mutationRate = ($this->epochsWithoutImprovement > 10) ? 25 : 10;
        
        // Стандартное обучение с адаптивной мутацией
        $this->trainWithMutation($input, $expected, $mutationRate);
        
        // Отслеживание прогресса
        if ($currentScore < $this->lastBestScore) {
            $this->epochsWithoutImprovement = 0;
            $this->lastBestScore = $currentScore;
        } else {
            $this->epochsWithoutImprovement++;
        }
    }
}
```

### Стратегии селекции

```php
// Элитизм + разнообразие
public function Train(string $input, string $expected): void {
    // Сохраняем топ 10% особей
    $elite = $this->selectElite(0.1);
    
    // Остальные 90% - потомки от элиты
    $this->networks = $elite;
    while (count($this->networks) < $this->populationSize) {
        $parent = $elite[rand(0, count($elite) - 1)];
        $this->networks[] = $parent->Born(15);
    }
}
```

## 🔬 Анализ алгоритма

### Сходимость

**Теоретические гарантии:**
- ✅ **Монотонность** - лучшая особь не ухудшается (элитизм)
- ✅ **Исследование** - мутации обеспечивают разнообразие
- ⚠️ **Локальные минимумы** - возможно застревание без достаточной мутации

**Практические наблюдения:**
- Быстрая сходимость на простых задачах (XOR, линейные функции)
- Стабильная работа на зашумленных данных
- Требует настройки параметров для сложных задач

### Вычислительная сложность

**Временная сложность на итерацию:**
- `O(P × N × L)` где:
  - `P` = размер популяции
  - `N` = размер входных данных  
  - `L` = количество слоев сети

**Пространственная сложность:**
- `O(P × W)` где `W` = общее количество весов в сети

## 🎛️ Продвинутые техники

### Многокритериальная оптимизация

```php
// Комбинирование метрик качества
private static function MultiObjectiveTest(
    Network $network, 
    string $input, 
    string $expected
): float {
    $prediction = (new Predictor($network))->Predict($input);
    
    // Точность предсказания
    $accuracy = 1.0 - manhattan_distance($prediction, $expected);
    
    // Сложность сети (регуляризация)
    $complexity = $network->getComplexity(); // Количество весов
    
    // Комбинированная метрика
    return $accuracy - 0.001 * $complexity;
}
```

### Островная эволюция

```php
class IslandTrainer {
    private array $islands = []; // Несколько изолированных популяций
    
    public function Train(string $input, string $expected): void {
        // Эволюция каждого острова независимо
        foreach ($this->islands as $island) {
            $island->evolve($input, $expected);
        }
        
        // Периодическая миграция лучших особей
        if ($this->generation % 10 == 0) {
            $this->migrate();
        }
    }
}
```

## 🧪 Отладка и диагностика

### Мониторинг разнообразия

```php
public function getPopulationDiversity(): float {
    $totalDistance = 0;
    $comparisons = 0;
    
    for ($i = 0; $i < count($this->networks); $i++) {
        for ($j = $i + 1; $j < count($this->networks); $j++) {
            $distance = $this->networks[$i]->getWeightDistance($this->networks[$j]);
            $totalDistance += $distance;
            $comparisons++;
        }
    }
    
    return $comparisons > 0 ? $totalDistance / $comparisons : 0;
}
```

### Визуализация сходимости

```php
public function logTrainingProgress(string $input, string $expected): void {
    $scores = [];
    foreach ($this->networks as $i => $network) {
        $scores[] = self::Test($network, $input, $expected);
    }
    
    echo "Generation: " . $this->generation . "\n";
    echo "Best: " . min($scores) . "\n";
    echo "Average: " . array_sum($scores) / count($scores) . "\n";
    echo "Worst: " . max($scores) . "\n";
    echo "Diversity: " . $this->getPopulationDiversity() . "\n\n";
}
```

## 📈 Рекомендации по использованию

### Выбор параметров

| Тип задачи | Размер популяции | Мутация | Критерий остановки |
|------------|------------------|---------|-------------------|
| XOR (простая) | 10-20 | 5-10% | error < 0.01 |
| Классификация | 50-100 | 10-20% | error < 0.05 |
| Аппроксимация | 100-200 | 15-25% | error < 0.1 |
| Сложные паттерны | 200-500 | 20-30% | 1000+ поколений |

### Лучшие практики

1. **Начинайте с малых популяций** для быстрого прототипирования
2. **Мониторьте разнообразие** - добавляйте мутации при застое
3. **Используйте элитизм** - сохраняйте лучшие решения
4. **Настраивайте мутацию** под сложность задачи
5. **Комбинируйте метрики** для многокритериальной оптимизации

### Типичные ошибки

- ❌ **Слишком малая популяция** → преждевременная сходимость
- ❌ **Слишком большая мутация** → случайный поиск
- ❌ **Слишком малая мутация** → застревание в локальных минимумах
- ❌ **Отсутствие мониторинга** → непонимание процесса обучения

## 🔮 Перспективы развития

### Планируемые улучшения

1. **Адаптивные операторы** - автоматическая настройка параметров
2. **Гибридные алгоритмы** - комбинация с backpropagation
3. **Параллельные вычисления** - GPU ускорение оценки популяции
4. **Продвинутая селекция** - турнирная, рулетка, ранговая
5. **Кроссовер операторы** - комбинирование весов родителей

### Интеграция с другими методами

```php
// Пример гибридного подхода
class HybridTrainer extends Trainer {
    public function Train(string $input, string $expected): void {
        // 1. Генетическая предварительная оптимизация
        parent::Train($input, $expected);
        
        // 2. Локальная доводка лучшей сети
        $this->best_network = $this->gradientDescent($this->best_network, $input, $expected);
    }
}
```

---

*Эта документация описывает текущую реализацию генетического алгоритма в библиотеке NiNa. Алгоритм постоянно развивается и улучшается.*