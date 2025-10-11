# Genetic Algorithm Training in NiNa

## 📋 Overview

The NiNa library uses a **genetic algorithm** to train neural networks. This evolutionary approach mimics natural selection to optimize neural network weights.

## 🧬 Genetic Algorithm Principles

### Core Concept

The genetic algorithm works with a **population** of neural networks, where each network represents an "individual". The algorithm iteratively improves the population through processes analogous to evolution:

1. **Initialization** - creating the initial population
2. **Fitness evaluation** - testing each individual
3. **Selection** - choosing the best individuals
4. **Reproduction** - creating offspring with mutations
5. **Replacement** - forming a new population

### Advantages of the Approach

- ✅ **No gradients required** - works with any activation functions
- ✅ **Global optimization** - avoids local minima
- ✅ **Simple implementation** - clear and intuitive algorithm
- ✅ **Noise resistance** - works with noisy data
- ✅ **Parallelizable** - each individual can be evaluated independently

## 🏗️ Implementation Architecture

### `Trainer` Class

The main class for genetic training. Manages the population of networks and the evolution process.

```php
class Trainer {
    private array $networks = [];      // Population of networks
    private Network $best_network;     // Best network
    
    public function __construct(Network $network, int $population_size);
    public function Train(string $input, string $expected): void;
    public function getBestNetwork(): Network;
}
```

### Supporting Components

- **`Network::Born()`** - creating offspring with mutations
- **`Layer::Born()`** - layer mutation
- **`mutate()`** - weight mutation function
- **`manhattan_distance()`** - fitness evaluation function

## 🔄 Detailed Algorithm

### 1. Population Initialization

```php
public function __construct(Network $network, int $pull) {
    $this->best_network = $network;
    
    // Create population of mutated base network copies
    for ($i = 0; $i < $pull; ++$i) {
        $this->networks[] = $network->Born(5); // 5% mutation
    }
}
```

**Process:**
1. Base network is accepted as "parent"
2. Population of specified size is created (usually 10-100 individuals)
3. Each individual is a mutated copy of the base network
4. Base network is saved as current best

### 2. Fitness Function (Quality Evaluation)

```php
private static function Test(Network $network, string $input, string $expected): float {
    $predictor = new Predictor($network);
    $calculated = $predictor->Predict($input);
    
    // Use Manhattan distance as error measure
    return manhattan_distance($calculated, $expected);
}
```

**Quality Metric:**
- **Manhattan distance** between prediction and expected result
- **Lower value = better individual**
- Formula: `∑|predicted[i] - expected[i]|`

### 3. Main Training Loop

```php
public function Train(string $input, string $expected): void {
    $best = $this->best_network;
    $bestScore = self::Test($best, $input, $expected);
    
    // Test entire population
    foreach ($this->networks as $network) {
        $score = self::Test($network, $input, $expected);
        if ($score < $bestScore) {
            $best = $network;
            $bestScore = $score;
        }
    }
    
    // Update best network
    $this->best_network = $best;
    
    // Create new population from best network
    $this->networks = [];
    for ($i = 0; $i < count($this->networks); ++$i) {
        $this->networks[] = $this->best_network->Born(10); // 10% mutation
    }
}
```

**Step-by-step Process:**

1. **Population evaluation** - each network is tested on training data
2. **Best selection** - individual with minimum error is chosen
3. **Elitism** - best individual becomes new "parent"
4. **Offspring generation** - new population is created from best network
5. **Mutations** - each offspring receives random weight changes

## 🧬 Mutation Mechanism

### Creating Offspring (`Network::Born`)

```php
public function Born(int $percent): Network {
    $layers = [];
    foreach ($this->layers as $layer) {
        $layers[] = $layer->Born($percent); // Mutate each layer
    }
    return new Network($layers, uuid());
}
```

### Layer Mutation (`Layer::Born`)

```php
public function Born(int $percent): Layer {
    $cells = [];
    foreach ($this->cells as $cell) {
        $newWeights = [];
        foreach ($cell->getWeights() as $weight) {
            $newWeights[] = mutate($weight, $percent); // Weight mutation
        }
        $cells[] = new Cell($cell->getName() . "_child", $newWeights, $cell->getActivationFunction());
    }
    return new Layer($cells, uuid());
}
```

### Weight Mutation Function

```php
function mutate(float $value, int $precision = 0): float {
    $rand = random_int(0, 100);
    if($rand < $precision) {
        $value += 0.001 * rand(-100, 100); // ±0.1 change
    }
    return limit($value); // Limit to [0, 1]
}
```

**Mutation Parameters:**
- **`$precision`** - probability of each weight mutation (0-100%)
- **Mutation strength** - random change ±0.1
- **Constraints** - weights remain in range [0, 1]

## 📊 Algorithm Parameters

### Population Size

```php
$trainer = new Trainer($network, $populationSize);
```

**Recommendations:**
- **Small tasks:** 10-20 individuals
- **Medium tasks:** 50-100 individuals  
- **Complex tasks:** 100-500 individuals

**Trade-offs:**
- ✅ **More individuals** = better exploration of solution space
- ❌ **More individuals** = more computations per iteration

### Mutation Level

```php
// In constructor - low mutation for diversity
$network->Born(5);  // 5% mutation probability

// In Train - more aggressive mutation
$network->Born(10); // 10% mutation probability
```

**Mutation Strategies:**
- **Conservative (1-5%)** - fine-tuning weights
- **Moderate (10-20%)** - balance exploration/exploitation
- **Aggressive (30-50%)** - broad space exploration

## 🎯 Practical Examples

### Example 1: Simple Training

```php
// Create base network
$network = Network::Create([10, 5, 1]); // 10 inputs, 5 hidden, 1 output

// Initialize trainer with population of 20 individuals
$trainer = new Trainer($network, 20);

// Train on one example
$trainer->Train("hello", "world");

// Get best network
$bestNetwork = $trainer->getBestNetwork();
```

### Example 2: Batch Training

```php
$trainer = new Trainer($network, 50);

$trainingData = [
    ["input1", "output1"],
    ["input2", "output2"], 
    ["input3", "output3"]
];

// Multiple training epochs
for ($epoch = 0; $epoch < 100; $epoch++) {
    foreach ($trainingData as [$input, $expected]) {
        $trainer->Train($input, $expected);
    }
}

$bestNetwork = $trainer->getBestNetwork();
```

### Example 3: Progress Monitoring

```php
$trainer = new Trainer($network, 30);

for ($epoch = 0; $epoch < 50; $epoch++) {
    foreach ($trainingData as [$input, $expected]) {
        $trainer->Train($input, $expected);
    }
    
    // Check progress
    $bestNetwork = $trainer->getBestNetwork();
    $predictor = new Predictor($bestNetwork);
    $prediction = $predictor->Predict($testInput);
    $error = manhattan_distance($prediction, $testExpected);
    
    echo "Epoch {$epoch}: Error = {$error}\n";
    
    // Stopping criterion
    if ($error < 0.01) {
        echo "Target accuracy achieved!\n";
        break;
    }
}
```

## ⚡ Performance Optimization

### Parameter Tuning

```php
// Adaptive mutation
class AdaptiveTrainer extends Trainer {
    private int $epochsWithoutImprovement = 0;
    private float $lastBestScore = PHP_FLOAT_MAX;
    
    public function Train(string $input, string $expected): void {
        $currentScore = $this->getBestScore();
        
        // Increase mutation during stagnation
        $mutationRate = ($this->epochsWithoutImprovement > 10) ? 25 : 10;
        
        // Standard training with adaptive mutation
        $this->trainWithMutation($input, $expected, $mutationRate);
        
        // Track progress
        if ($currentScore < $this->lastBestScore) {
            $this->epochsWithoutImprovement = 0;
            $this->lastBestScore = $currentScore;
        } else {
            $this->epochsWithoutImprovement++;
        }
    }
}
```

### Selection Strategies

```php
// Elitism + diversity
public function Train(string $input, string $expected): void {
    // Keep top 10% individuals
    $elite = $this->selectElite(0.1);
    
    // Remaining 90% - offspring from elite
    $this->networks = $elite;
    while (count($this->networks) < $this->populationSize) {
        $parent = $elite[rand(0, count($elite) - 1)];
        $this->networks[] = $parent->Born(15);
    }
}
```

## 🔬 Algorithm Analysis

### Convergence

**Theoretical Guarantees:**
- ✅ **Monotonicity** - best individual doesn't worsen (elitism)
- ✅ **Exploration** - mutations ensure diversity
- ⚠️ **Local minima** - possible stagnation without sufficient mutation

**Practical Observations:**
- Fast convergence on simple tasks (XOR, linear functions)
- Stable work with noisy data
- Requires parameter tuning for complex tasks

### Computational Complexity

**Time complexity per iteration:**
- `O(P × N × L)` where:
  - `P` = population size
  - `N` = input data size  
  - `L` = number of network layers

**Space complexity:**
- `O(P × W)` where `W` = total number of weights in network

## 🎛️ Advanced Techniques

### Multi-objective Optimization

```php
// Combining quality metrics
private static function MultiObjectiveTest(
    Network $network, 
    string $input, 
    string $expected
): float {
    $prediction = (new Predictor($network))->Predict($input);
    
    // Prediction accuracy
    $accuracy = 1.0 - manhattan_distance($prediction, $expected);
    
    // Network complexity (regularization)
    $complexity = $network->getComplexity(); // Number of weights
    
    // Combined metric
    return $accuracy - 0.001 * $complexity;
}
```

### Island Evolution

```php
class IslandTrainer {
    private array $islands = []; // Multiple isolated populations
    
    public function Train(string $input, string $expected): void {
        // Evolve each island independently
        foreach ($this->islands as $island) {
            $island->evolve($input, $expected);
        }
        
        // Periodic migration of best individuals
        if ($this->generation % 10 == 0) {
            $this->migrate();
        }
    }
}
```

## 🧪 Debugging and Diagnostics

### Diversity Monitoring

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

### Convergence Visualization

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

## 📈 Usage Recommendations

### Parameter Selection

| Task Type | Population Size | Mutation | Stopping Criterion |
|-----------|----------------|----------|-------------------|
| XOR (simple) | 10-20 | 5-10% | error < 0.01 |
| Classification | 50-100 | 10-20% | error < 0.05 |
| Approximation | 100-200 | 15-25% | error < 0.1 |
| Complex patterns | 200-500 | 20-30% | 1000+ generations |

### Best Practices

1. **Start with small populations** for rapid prototyping
2. **Monitor diversity** - add mutations during stagnation
3. **Use elitism** - preserve best solutions
4. **Tune mutation** to task complexity
5. **Combine metrics** for multi-objective optimization

### Common Mistakes

- ❌ **Too small population** → premature convergence
- ❌ **Too high mutation** → random search
- ❌ **Too low mutation** → stuck in local minima
- ❌ **No monitoring** → lack of understanding of training process

## 🔮 Future Developments

### Planned Improvements

1. **Adaptive operators** - automatic parameter tuning
2. **Hybrid algorithms** - combination with backpropagation
3. **Parallel computing** - GPU acceleration of population evaluation
4. **Advanced selection** - tournament, roulette, rank-based
5. **Crossover operators** - combining parent weights

### Integration with Other Methods

```php
// Example hybrid approach
class HybridTrainer extends Trainer {
    public function Train(string $input, string $expected): void {
        // 1. Genetic pre-optimization
        parent::Train($input, $expected);
        
        // 2. Local fine-tuning of best network
        $this->best_network = $this->gradientDescent($this->best_network, $input, $expected);
    }
}
```

---

*This documentation describes the current implementation of the genetic algorithm in the NiNa library. The algorithm is continuously being developed and improved.*