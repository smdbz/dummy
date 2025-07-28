# boot_dummy

A Python package for generating synthetic data with realistic statistical properties.

## Installation

```bash
pip install boot_dummy
```

## Features

- Generate synthetic data based on statistical properties of real datasets
- Support for various data types (numeric, datetime, categorical, text)
- Preserve statistical distributions and relationships from original data

## Usage

### Basic Usage

```python
import pandas as pd
from boot_dummy import GenerateData

# Load your existing dataset
original_data = pd.read_csv('your_dataset.csv')

# Create a generator from the original data
generator = GenerateData(original_data)

# Generate synthetic data with the same properties
synthetic_data = generator.generate_dataframe(n_rows=1000)

# Save or use the synthetic data
synthetic_data.to_csv('synthetic_data.csv', index=False)
```

### Generate Individual Rows

```python
# Generate a single row of synthetic data
single_row = generator.generate_row()
print(single_row)
```

## How It Works

The package analyzes your original dataset to determine:
- Data types (numeric, datetime, categorical, text)
- Statistical properties (mean, standard deviation, min/max values)
- Value distributions and relationships

It then generates new data that preserves these properties while creating entirely synthetic records.

## Requirements

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0

## License

MIT