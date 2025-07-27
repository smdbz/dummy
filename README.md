# boot-dummy

A Python package to create synthetic data with realistic statistical properties.

## Overview

boot-dummy allows you to:

1. Generate synthetic customer data with predefined rules
2. Load datasets from various file formats (CSV, Excel, JSON, Parquet)
3. Generate enhanced descriptive statistics for datasets
4. Create synthetic data that matches the statistical properties of existing datasets

## Installation

```bash
pip install boot_dummy
```

## Quick Start

### Generate synthetic customer data

```python
from boot_dummy import GenerateData

# Generate 1000 customer records
generator = GenerateData(num_records=1000)
customer_data = generator.generate()

# View the data
print(customer_data.head())
```

### Generate data based on an existing dataset

```python
from boot_dummy import GenerateData
import pandas as pd

# Option 1: Using a file path
result = GenerateData.generate_from_dataset(
    dataset_path='your_dataset.csv',
    num_records=500
)

# Option 2: Using an existing DataFrame
df = pd.read_csv('your_dataset.csv')
result = GenerateData.generate_from_dataset(
    dataset=df,
    num_records=500
)

# Access the components
original_data = result['original_data']
statistics = result['stats']
synthetic_data = result['synthetic_data']
```

## Workflow

The typical workflow with boot-dummy is:

1. **Load or generate initial data**
   - Use `GenerateData.generate()` to create synthetic customer data, or
   - Use `GenerateData.load_dataset()` to load data from a file

2. **Analyze the data**
   - Use `GenerateData.enhanced_describe()` to get detailed statistics

3. **Generate new data based on statistics**
   - Use `GenerateData.generate_from_stats()` to create new data with similar properties

4. **All-in-one approach**
   - Use `GenerateData.generate_from_dataset()` to perform all steps in one call

## Examples

See the [examples](./examples) directory for complete usage examples.

## Resources

This package was inspired by and borrows from:
* [Real Python - Publish Package](https://realpython.com/pypi-publish-python-package/#get-to-know-python-packaging)
* [Data Camp - Synthetic Data Generation](https://www.datacamp.com/tutorial/synthetic-data-generation)
