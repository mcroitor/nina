# NiNa Library Architecture

## 🏗️ Architecture Overview

NiNa is built on object-oriented programming principles with clear separation of responsibilities between components. The library uses a hierarchical structure from individual neurons to complete neural networks.

## 📐 Architecture Diagram

```txt
┌─────────────────────────────────────────────────────────────┐
│                     NiNa Neural Network                     │
├─────────────────────────────────────────────────────────────┤
│                        Network                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Layer    │  │    Layer    │  │    Layer    │         │
│  │   Input     │─▶│   Hidden    │─▶│   Output    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Cell     │  │    Cell     │  │    Cell     │         │
│  │  (Neuron)   │  │  (Neuron)   │  │  (Neuron)   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Activation  │  │ Activation  │  │ Activation  │         │
│  │  Function   │  │  Function   │  │  Function   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘

Training & Prediction Components:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Trainer   │    │  Predictor  │    │ Exceptions  │
│ (Genetic    │    │  (Forward   │    │   System    │
│ Algorithm)  │    │   Pass)     │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🧩 Core Components

### 1. Cell (Neuron) - `mc\nina\Cell`

**Responsibility:** Individual neural processing unit

**Key Features:**

- Stores weights (DNA)
- Applies activation function
- Performs forward propagation
- Supports mutation for genetic algorithm

**Dependencies:**

- ActivationFunction interface
- Utility functions from lib.php

**Data Flow:**

```txt
Input Array → Weight Multiplication → Sum → Activation Function → Output
```

**Design Patterns:**

- **Strategy Pattern**: For activation function selection
- **Factory Pattern**: Cell::Create() static method
- **Serialization**: JSON save/load capability

### 2. Layer - `mc\nina\Layer`

**Responsibility:** Collection of neurons working in parallel

**Key Features:**

- Manages array of Cell objects
- Parallel processing of inputs
- Layer-level mutation
- Uniform interface for network integration

**Dependencies:**

- Cell objects
- Exception system for validation

**Data Flow:**

```txt
Input Array → [Cell1, Cell2, ..., CellN] → Output Array
```

**Design Patterns:**

- **Composite Pattern**: Collection of cells behaving as single unit
- **Iterator Pattern**: Access to individual cells

### 3. Network - `mc\nina\Network`

**Responsibility:** Complete neural network architecture

**Key Features:**

- Sequential layer processing
- Automatic input size adjustment
- Network-level mutation
- Complete forward propagation

**Dependencies:**

- Layer objects
- Exception system for architecture validation

**Data Flow:**

```txt
Input → Layer1 → Layer2 → ... → LayerN → Output
```

**Design Patterns:**

- **Chain of Responsibility**: Sequential layer processing
- **Builder Pattern**: Network::Create() for standard architectures
- **Facade Pattern**: Simplified interface for complex operations

### 4. Trainer - `mc\nina\Trainer`

**Responsibility:** Genetic algorithm implementation for network training

**Key Features:**

- Population management (multiple networks)
- Fitness evaluation using Manhattan distance
- Selection and mutation operations
- Evolutionary optimization

**Dependencies:**

- Network objects
- Utility functions for distance calculation

**Algorithm:**

```txt
1. Initialize population of networks
2. Evaluate fitness for each network
3. Select best performing network
4. Generate new population based on best network
5. Apply mutations
6. Repeat from step 2
```

**Design Patterns:**

- **Template Method**: Standard genetic algorithm template
- **Observer Pattern**: Fitness evaluation and selection

### 5. Predictor - `mc\nina\Predictor`

**Responsibility:** Interface for making predictions with trained networks

**Key Features:**

- Simple prediction interface
- String to array conversion
- Result formatting
- Error handling for predictions

**Dependencies:**

- Trained Network object
- String/array conversion utilities

**Data Flow:**

```txt
String Input → Array Conversion → Network Processing → String Output
```

**Design Patterns:**

- **Adapter Pattern**: Converts string inputs to network-compatible arrays
- **Proxy Pattern**: Simplified interface to network functionality

## 🎯 Activation Function System

### Interface Design - `ActivationFunction`

**Purpose:** Unified interface for mathematical functions

**Methods:**

- `activate(float $input): float` - Apply activation function
- `derivative(float $input): float` - Calculate derivative (for future backprop)
- `getName(): string` - Function identification

### Available Implementations:

#### ReLU (Rectified Linear Unit)

- **Formula:** f(x) = max(0, x)
- **Use Case:** Hidden layers, sparse activation
- **Advantage:** Fast computation, no vanishing gradient

#### Sigmoid

- **Formula:** f(x) = 1 / (1 + e^(-x))
- **Use Case:** Binary classification, probability output
- **Advantage:** Smooth gradient, bounded output

#### Tanh (Hyperbolic Tangent)

- **Formula:** f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **Use Case:** Hidden layers, centered output
- **Advantage:** Zero-centered output

#### Softmax

- **Formula:** f(x_i) = e^(x_i) / Σ(e^(x_j))
- **Use Case:** Multi-class classification
- **Advantage:** Probability distribution output

## 🛡️ Exception System Architecture

### Hierarchical Design

```txt
NinaException (Abstract Base)
├── ValidationException (6001-6099)
├── NetworkException (1001-1099)
├── LayerException (2001-2099)
├── TrainingException (3001-3099)
├── SerializationException (4001-4099)
└── ComputationException (5001-5099)
```

### Exception Features:

**Context Management:**

- Detailed error context
- Parameter values that caused errors
- Operation being performed

**Severity Levels:**

- `error`: Standard recoverable errors
- `warning`: Non-critical issues
- `critical`: System-threatening errors

**Debugging Support:**

- Detailed error messages
- Stack trace preservation
- JSON serialization for logging

## 🔄 Data Flow Architecture

### Training Flow (Genetic Algorithm)

```txt
1. Input Data (String)
   ↓
2. Population Evaluation
   ├── Network 1 → Prediction → Fitness Score
   ├── Network 2 → Prediction → Fitness Score
   └── Network N → Prediction → Fitness Score
   ↓
3. Best Network Selection (Manhattan Distance)
   ↓
4. Population Generation
   ├── Copy Best Network
   ├── Apply Mutations (10% variation)
   └── Fill Population
   ↓
5. Next Generation Ready
```

### Prediction Flow

```txt
1. String Input
   ↓
2. String to Array Conversion
   ↓
3. Network Forward Pass
   ├── Layer 1 Processing
   ├── Layer 2 Processing
   └── Layer N Processing
   ↓
4. Array to String Conversion
   ↓
5. String Output
```

### Serialization Flow

```txt
Save:
Network → Layers Array → Cells Array → JSON String → File

Load:
File → JSON String → Cells Array → Layers Array → Network
```

## 📏 SOLID Principles Implementation

### Single Responsibility Principle (SRP)

- **Cell**: Only neuron computation
- **Layer**: Only layer management
- **Network**: Only network architecture
- **Trainer**: Only training logic
- **Predictor**: Only prediction interface

### Open/Closed Principle (OCP)

- **ActivationFunction interface**: New functions without modifying existing code
- **Exception system**: New exception types without breaking existing handlers

### Liskov Substitution Principle (LSP)

- **All ActivationFunction implementations**: Interchangeable
- **Exception hierarchy**: All exceptions can be caught as NinaException

### Interface Segregation Principle (ISP)

- **ActivationFunction**: Small, focused interface
- **Separate interfaces**: No forced dependencies on unused methods

### Dependency Inversion Principle (DIP)

- **Cell depends on ActivationFunction interface**: Not concrete implementations
- **High-level modules**: Don't depend on low-level details

## 🔧 Design Patterns Summary

### Creational Patterns

- **Factory Method**: Cell::Create(), Layer::Create(), Network::Create()
- **Builder**: Network construction with fluent interface potential

### Structural Patterns

- **Composite**: Layer containing multiple Cells
- **Adapter**: Predictor adapting string interface to network arrays
- **Facade**: Network providing simple interface to complex layer operations

### Behavioral Patterns

- **Strategy**: ActivationFunction selection
- **Template Method**: Genetic algorithm structure in Trainer
- **Chain of Responsibility**: Sequential layer processing
- **Observer**: Potential for training progress monitoring

## 🚀 Scalability and Performance

### Current Architecture Strengths:

- **Modular Design**: Easy to extend and modify
- **Clear Interfaces**: Well-defined component boundaries
- **Exception Safety**: Robust error handling
- **Serialization**: Model persistence and sharing

### Future Optimization Opportunities:

- **Batch Processing**: Multiple inputs simultaneously
- **GPU Acceleration**: Parallel computation support
- **Advanced Algorithms**: Backpropagation, gradient descent
- **Memory Optimization**: Large network handling

### Extensibility Points:

- **New Activation Functions**: Implement ActivationFunction interface
- **Training Algorithms**: Alternative to genetic approach
- **Layer Types**: Convolutional, LSTM, attention layers
- **Optimization**: Learning rate, momentum, regularization

## 📈 Performance Considerations

### Time Complexity:

- **Cell.Calculate()**: O(n) where n = number of weights
- **Layer.Calculate()**: O(m*n) where m = cells, n = input size
- **Network.Calculate()**: O(l*m*n) where l = layers, m = cells per layer
- **Training**: O(p*g*l*m*n) where p = population, g = generations

### Memory Usage:

- **Cell**: O(n) for weights storage
- **Layer**: O(m*n) for all cells
- **Network**: O(l*m*n) for complete architecture
- **Trainer**: O(p*l*m*n) for population storage

### Optimization Strategies:

- **Weight Sharing**: Reduce memory in similar layers
- **Lazy Loading**: Load network parts on demand
- **Caching**: Cache frequently used calculations
- **Parallel Processing**: Multi-threaded layer computation

---

*This architecture documentation reflects NiNa v0.2.0 and provides foundation for future enhancements.*
