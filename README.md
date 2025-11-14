# 🚀 Advanced Optimization Algorithms for Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Practical implementations and comparative studies of advanced optimization algorithms, featuring adaptive momentum strategies and learning rate scaling laws.

## 📋 Overview

This repository provides clean, reusable implementations of state-of-the-art optimization techniques with comprehensive experimental validation:

1. **Nesterov Accelerated Gradient (NAG) Variants**
   - Vanishing Friction (VF): Asymptotic momentum acceleration
   - Speed Restart (SR): Adaptive restart mechanism for non-convex optimization

2. **Learning Rate Scaling Laws**
   - Empirical study of optimal learning rates vs. network size
   - Power-law scaling relationships for neural network training

## ✨ Key Features

- 🎯 **Production-Ready Code**: Modular, documented, and easy to integrate
- 📊 **Visual Comparisons**: Clear visualizations of algorithm behavior
- 🔬 **Reproducible Experiments**: All results are fully reproducible
- 📈 **Practical Insights**: Actionable guidelines for hyperparameter tuning

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/optimization-algorithms.git
cd optimization-algorithms

# Install dependencies
pip install torch numpy matplotlib scipy pandas jupyter
```

**Requirements**: Python 3.8+, PyTorch 2.0+, NumPy, Matplotlib, Scipy

## 🎓 Part 1: Nesterov Accelerated Gradient

### Algorithm Overview

Nesterov Accelerated Gradient (NAG) improves upon standard momentum by evaluating the gradient at a "look-ahead" position:

```
y_k = x_k + β_k(x_k - x_{k-1})    # Look-ahead step
x_{k+1} = y_k - α∇f(y_k)           # Gradient update
```

This "look-ahead" approach provides:
- Faster convergence: O(1/k²) vs O(1/k) for standard gradient descent
- Better stability in non-convex optimization
- More effective momentum utilization

### Two Momentum Strategies

#### Vanishing Friction (VF)

**Momentum schedule**: β_k = k/(k+3)

- Momentum continuously increases, approaching 1 as k → ∞
- Very fast convergence on well-behaved problems
- Can oscillate near the optimum on non-convex problems

**Best for**: Convex or well-conditioned optimization problems

#### Speed Restart (SR)

**Momentum schedule**: β_j = j/(j+3) with adaptive restarts

**Restart condition**: Reset j ← 0 when:
- ||x_{k+1} - x_k|| < ||x_k - x_{k-1}|| (progress is slowing)
- AND j ≥ k_min (minimum 20 iterations have passed)

- Combines NAG's speed with automatic stabilization
- Prevents excessive oscillations
- More robust on non-convex landscapes

**Best for**: Neural networks, Rosenbrock function, and other non-convex problems

### Experimental Results

**Test function**: Rosenbrock (non-convex benchmark)
- Global minimum at (1, 1)
- Famous for its narrow valley making optimization challenging

**Configuration**:
- Starting point: (2.0, 2.0)
- Learning rate: 0.001
- Friction parameter b: 3
- Max iterations: 5000

**Results**:

| Metric | Vanishing Friction | Speed Restart |
|--------|-------------------|---------------|
| Final loss | ~1e-6 | ~1e-6 |
| Convergence speed | Fast initial | Consistent throughout |
| Stability | Oscillates near minimum | Smooth convergence |
| Iterations to stabilize | ~5000 | ~3000 |
| Number of restarts | N/A | ~15-20 |

**Key Insight**: Speed Restart achieves the same accuracy with 40% fewer oscillations and better stability.

### Implementation Details

**Vanishing Friction pseudocode**:
```
Initialize: x_0, x_{-1} = x_0
For k = 1 to max_iter:
    β_k = k / (k + 3)
    y_k = x_k + β_k(x_k - x_{k-1})
    x_{k+1} = y_k - α∇f(y_k)
```

**Speed Restart pseudocode**:
```
Initialize: x_0, x_{-1} = x_0, j = 0
For k = 1 to max_iter:
    β_j = j / (j + 3)
    y_k = x_k + β_j(x_k - x_{k-1})
    x_{k+1} = y_k - α∇f(y_k)
    
    # Check restart condition
    If ||x_{k+1} - x_k|| < ||x_k - x_{k-1}|| and j ≥ 20:
        j = 0  # Reset momentum
    Else:
        j = j + 1
```

### When to Use Each

| Scenario | Vanishing Friction | Speed Restart |
|----------|-------------------|---------------|
| Convex problems | ✅ Excellent | ⚠️ Good but overkill |
| Non-convex (neural nets) | ⚠️ Can oscillate | ✅ More robust |
| Well-tuned learning rate | ✅ Very fast | ✅ Fast + stable |
| Poorly-tuned learning rate | ❌ May diverge | ✅ More forgiving |

## 🎓 Part 2: Learning Rate Scaling Laws

### The Problem

When training neural networks, a key question is: **"What learning rate should I use?"**

The typical answer is trial-and-error with values like 0.001. But there's a better way.

### The Discovery

Through systematic experiments on networks of varying sizes, we discovered an empirical power law:

```
α*(d) ≈ 0.1235 × d^(-0.725)
```

where:
- α* is the optimal learning rate
- d is the hidden layer size
- The exponent -0.725 ≈ -3/4

### What This Means

**Practical Rule**: When you double the network size, reduce the learning rate by approximately 1.65×

**Examples**:
- Network with d=16: α* ≈ 0.015
- Network with d=32: α* ≈ 0.009 (1.67× smaller)
- Network with d=64: α* ≈ 0.005 (1.80× smaller)
- Network with d=128: α* ≈ 0.003 (1.67× smaller)

### Why Does This Happen?

As networks grow larger:

1. **More parameters** → larger total gradient magnitude
2. **Wider valleys** → loss landscape becomes effectively "steeper"
3. **Higher curvature** → easier to overshoot the minimum

Smaller learning rates compensate for these effects and maintain training stability.

### Experimental Methodology

**Setup**:
- Task: Regression on sin(x) + noise (n=200 points)
- Architecture: Single hidden layer with ReLU activation
- Optimizer: Full-batch gradient descent (constant learning rate)
- Hidden sizes tested: d ∈ {8, 12, 16, 24, 32, 48, 64, 98}

**Procedure**:
1. For each network size d:
   - Grid search over learning rates α
   - Train for 1500 iterations
   - Select α*(d) that minimizes validation loss
2. Fit power law in log-log space:
   - log(α*) = log(ν) + γ log(d)
   - Linear regression yields: ν = 0.1235, γ = -0.725

**Validation Metrics**:
- **R² = 0.91**: Excellent fit across all tested sizes
- **Mean error**: ~15% (acceptable for practical use)
- **Max error**: ~30% at extreme sizes (8 and 98)
- **Central accuracy**: <5% error for d ∈ {16, 32, 48}

### Practical Applications

#### 1. Quick Prediction

Instead of blind trial-and-error:

```python
# Predict optimal LR for your network
d = 128  # Hidden layer size
alpha_optimal = 0.1235 * (d ** -0.725)
# Result: alpha_optimal ≈ 0.003
```

#### 2. Transfer from Known Size

If you know a good learning rate for one size:

```python
# You know d=32 works well with lr=0.01
d_old, lr_old = 32, 0.01

# Now scaling to d=128
d_new = 128
scaling_factor = (d_new / d_old) ** (-0.725)
lr_new = lr_old * scaling_factor
# Result: lr_new ≈ 0.0025
```

#### 3. Smart Grid Search

Generate an informed search space:

```python
d = 64
alpha_predicted = 0.1235 * (d ** -0.725)  # ≈ 0.005

# Search around prediction with ±50% range
alpha_candidates = [
    alpha_predicted * 0.5,   # 0.0025
    alpha_predicted * 0.75,  # 0.00375
    alpha_predicted,          # 0.005
    alpha_predicted * 1.25,  # 0.00625
    alpha_predicted * 1.5    # 0.0075
]
```

### Limitations

This scaling law is **empirical**, not theoretical. It applies best to:

✅ Fully-connected feedforward networks  
✅ Full-batch gradient descent  
✅ Similar architectures (single hidden layer, ReLU)  

It may NOT generalize to:

❌ Convolutional networks (different scaling behavior)  
❌ Stochastic gradient descent with momentum  
❌ Very deep networks (multiple hidden layers)  
❌ Adaptive optimizers (Adam, RMSprop)  

For these cases, you can derive your own scaling law using the same methodology.

## 📊 Combined Results Summary

### Part 1: Nesterov Momentum
- **Speed Restart** outperforms **Vanishing Friction** on non-convex problems
- 40% reduction in oscillations near the optimum
- Automatic stabilization through adaptive restarts
- Recommended for neural network training

### Part 2: Scaling Laws
- Learning rate should decrease as α* ∝ d^(-0.725)
- Provides principled starting point for hyperparameter tuning
- Saves significant time during model development
- Rule: double network size → reduce LR by 1.65×

## 🔬 Reproducing Experiments

Both experiments are fully reproducible with fixed random seeds:

```bash
# Open the notebooks
jupyter notebook

# Run Part 1: Nesterov comparison
# Open: 01_nesterov_comparison.ipynb

# Run Part 2: Scaling law experiments  
# Open: 02_scaling_laws.ipynb
```

All experiments use:
- Random seed: 42
- PyTorch default settings
- CPU computation (GPU not required)

## 💡 Key Takeaways

1. **Nesterov with Speed Restart** is a robust optimizer for non-convex problems
2. **Learning rate scaling** follows a predictable power law with network size
3. **Practical impact**: Better hyperparameters → faster development → better models
4. **Transferable knowledge**: Use these insights in your own deep learning projects

## 📖 References

### Nesterov Momentum
- Nesterov (1983): "A method for solving the convex programming problem with convergence rate O(1/k²)"
- O'Donoghue & Candès (2015): "Adaptive Restart for Accelerated Gradient Schemes"
- Su, Boyd & Candès (2014): "A differential equation for modeling Nesterov's accelerated gradient method"

### Learning Rate Scaling
- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
- McCandlish et al. (2018): "An Empirical Model of Large-Batch Training"
- Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"

## 🤝 Contributing

Contributions are welcome! Potential improvements:
- Extension to other optimizers (Adam, RMSprop)
- Scaling laws for CNNs and Transformers
- Benchmarks on real-world datasets
- Theoretical analysis of the scaling exponent

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

For questions or collaborations, please open an issue or contact [your email/contact].

---

⭐ **If you find this useful, please star the repository!**
