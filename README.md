# Hotelling Game Library (HTGlib)

A Python library for modeling Hotelling equilibria on city graphs with fast Voronoi domain recalculation.

## Features

- **Fast Voronoi Recalculation**: Efficient add/remove seller operations
- **Heuristic Optimization**: Stochastic search for better equilibria  
- **Multiple Profit Metrics**: Vertex count, edge length, demand weights
- **Interactive Visualization**: Plotly-based graphs with sliders
- **Graph Format Support**: GraphML, GML, edge lists
- **CLI Interface**: Command-line tools for analysis

## Quick Start

```python
from htglib import HotellingGame

# Load graph and place sellers
game = HotellingGame()
game.load_graph("samples/line_5.txt")
game.set_sellers([0, 4])

# Compute profits
profits = game.compute_profits()
print(f"Profits: {profits}")

# Run optimization with visualization
game.enable_visualization(True)
result = game.run_optimization(max_iterations=50)

# Create interactive plot
fig = game.plot_optimization_history()
game.save_visualization(fig, "optimization.html")
```

## Command Line Usage

```bash
# Basic analysis
python -m htglib.main --graph samples/line_5.txt --sellers 0,4 --stats

# Optimization with visualization
python -m htglib.main --graph samples/line_5.txt --sellers 0,4 \
    --optimize --visualize optimization.html --visualize-static before_after.html
```

## Visualization

The library provides three types of visualizations:

1. **Static Plots**: Current game state
2. **Interactive History**: Step-by-step optimization with slider
3. **Before/After**: Side-by-side comparison

```python
# Enable snapshot collection
game.enable_visualization(True)

# Run optimization
result = game.run_optimization(max_iterations=50)

# Create visualizations
fig_history = game.plot_optimization_history()
fig_comparison = game.plot_before_after()
```

## Installation

```bash
pip install -r requirements.txt
# Note: Requires graph-tool for graph operations
```

## Project Structure

```
htglib/
├── graph_model.py      # Graph loading and operations
├── distance_matrix.py  # Precomputed distances
├── ownership.py        # Voronoi domain management
├── update_engine.py    # Fast add/remove operations
├── profit.py          # Profit calculations
├── heuristics.py      # Optimization algorithms
├── visualization.py   # Plotly-based plotting
├── api.py            # Main HotellingGame class
└── main.py           # CLI interface
```

## API Reference

### Core Classes

- `HotellingGame`: Main interface class
- `GraphWrapper`: Graph representation
- `DistanceMatrix`: Precomputed distances
- `Ownership`: Voronoi domains
- `ProfitCalculator`: Profit computation
- `UpdateEngine`: Incremental updates
- `HeuristicEngine`: Optimization

### Key Methods

- `load_graph(path)`: Load graph file
- `set_sellers(list)`: Set seller positions
- `add_seller(v)` / `remove_seller(v)`: Dynamic operations
- `compute_profits()`: Calculate profits
- `run_optimization()`: Heuristic search
- `plot_*()`: Create visualizations
