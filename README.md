# boot_dummy

A Python package for generating synthetic data with realistic statistical properties.

**See it in action**: [Google colab notebook](https://colab.research.google.com/drive/1Td8_GPN0ses6Ts99rHyqgdPqcro0CnIx?usp=sharing)

This package was created for the 2025 [Boot.dev](https://www.boot.dev/) hackathon, the first hackathon I've participated in :D
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

## Does it suck?
When I first started the hackathon, I wanted to create a sort of "reverse data science" package, starting with data definitions and model weights to generate data.
You could then make predictions on the data and evaluate your model against known input parameters. What it turned into was "can I make any package that can make any dummy data?"

The dummy data that it outputs kinda sucks. That said, I’m super happy with the package design. The elements are modular and can be improved. The first version of the codebase was a bit smelly, and I’m glad I’ve learnt enough to recognise this.

In conclusion, I’ve never made a python package before, and this is the closest thing I’ve made to real software so I love it.

## Resources
- https://realpython.com/lessons/configure-python-package/
- https://www.datacamp.com/tutorial/synthetic-data-generation
- https://www.kaggle.com/learn
