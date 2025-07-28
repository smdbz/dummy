"""
Generate synthetic data with realistic statistical properties
"""

import numpy as np
import pandas as pd
import Dict, Union, Optional, Any

class GenerateData:
    """
    A class for generating synthetic data and analyzing datasets.
    
    This class provides functionality to generate synthetic data, analyze existing 
    datasets, and create new data that matches statistical properties of real data.
    """

    def __init__(self, seed_df: pd.DataFrame):
        """
        Initialize the DataGenerator with a seed DataFrame.

        Parameters
        ----------
        seed_df : pd.DataFrame
            The seed DataFrame to analyze and base synthetic data generation on.
        """
        self.seed_df = seed_df.copy()
        self.columns = seed_df.columns
        self.column_types = {}
        self.column_stats = {}
        self._analyze_columns()

    def _analyze_columns(self) -> None:
        """
        Analyze each column in the seed DataFrame to determine its type and statistics.
        """
        for col in self.columns:
            # Determine column type
            if pd.api.types.is_numeric_dtype(self.seed_df[col]):
                if self.seed_df[col].dtype == 'int64' or self.seed_df[col].dtype == 'int32':
                    self.column_types[col] = 'integer'
                else:
                    self.column_types[col] = 'float'

                # Calculate statistics for numeric columns
                self.column_stats[col] = {
                    'mean': self.seed_df[col].mean(),
                    'std': self.seed_df[col].std(),
                    'min': self.seed_df[col].min(),
                    'max': self.seed_df[col].max(),
                    'unique_values': self.seed_df[col].unique().tolist() if len(
                        self.seed_df[col].unique()) < 10 else None
                }

            elif pd.api.types.is_datetime64_dtype(self.seed_df[col]):
                self.column_types[col] = 'datetime'
                # Calculate statistics for datetime columns
                self.column_stats[col] = {
                    'min': self.seed_df[col].min(),
                    'max': self.seed_df[col].max(),
                    'unique_values': self.seed_df[col].unique().tolist() if len(
                        self.seed_df[col].unique()) < 10 else None
                }

            elif pd.api.types.is_categorical_dtype(self.seed_df[col]) or self.seed_df[col].nunique() / len(
                    self.seed_df) < 0.1:
                self.column_types[col] = 'categorical'
                # Calculate statistics for categorical columns
                value_counts = self.seed_df[col].value_counts(normalize=True)
                self.column_stats[col] = {
                    'categories': value_counts.index.tolist(),
                    'probabilities': value_counts.values.tolist()
                }

            else:
                self.column_types[col] = 'text'
                # Calculate statistics for text columns
                self.column_stats[col] = {
                    'unique_values': self.seed_df[col].unique().tolist(),
                    'length_mean': self.seed_df[col].str.len().mean(),
                    'length_std': self.seed_df[col].str.len().std()
                }

    def _generate_numeric_value(self, col: str) -> Union[int, float]:
        """
        Generate a synthetic numeric value based on the column's statistics.

        Parameters
        ----------
        col : str
            The column name to generate a value for.

        Returns
        -------
        Union[int, float]
            A synthetic numeric value.
        """
        stats = self.column_stats[col]

        # If we have a small set of unique values, sample from them
        if stats['unique_values'] is not None:
            return np.random.choice(stats['unique_values'])

        # Otherwise generate from a normal distribution with the same mean and std
        value = np.random.normal(stats['mean'], stats['std'])

        # Clip to min/max range
        value = max(stats['min'], min(stats['max'], value))

        # Convert to int if the column type is integer
        if self.column_types[col] == 'integer':
            value = int(round(value))

        return value

    def _generate_datetime_value(self, col: str) -> pd.Timestamp:
        """
        Generate a synthetic datetime value based on the column's statistics.

        Parameters
        ----------
        col : str
            The column name to generate a value for.

        Returns
        -------
        pd.Timestamp
            A synthetic datetime value.
        """
        stats = self.column_stats[col]

        # If we have a small set of unique values, sample from them
        if stats['unique_values'] is not None:
            return np.random.choice(stats['unique_values'])

        # Otherwise generate a random datetime between min and max
        min_ts = stats['min'].timestamp()
        max_ts = stats['max'].timestamp()
        random_ts = np.random.uniform(min_ts, max_ts)

        return pd.Timestamp.fromtimestamp(random_ts)

    def _generate_categorical_value(self, col: str) -> Any:
        """
        Generate a synthetic categorical value based on the column's statistics.

        Parameters
        ----------
        col : str
            The column name to generate a value for.

        Returns
        -------
        Any
            A synthetic categorical value.
        """
        stats = self.column_stats[col]
        return np.random.choice(stats['categories'], p=stats['probabilities'])

    def _generate_text_value(self, col: str) -> str:
        """
        Generate a synthetic text value based on the column's statistics.

        Parameters
        ----------
        col : str
            The column name to generate a value for.

        Returns
        -------
        str
            A synthetic text value.
        """
        stats = self.column_stats[col]

        # Sample from unique values
        return np.random.choice(stats['unique_values'])

    def generate_row(self) -> Dict[str, Any]:
        """
        Generate a single synthetic row based on the seed DataFrame.

        Returns
        -------
        Dict[str, Any]
            A dictionary representing a synthetic row.
        """
        row = {}

        for col in self.columns:
            if self.column_types[col] == 'integer' or self.column_types[col] == 'float':
                row[col] = self._generate_numeric_value(col)
            elif self.column_types[col] == 'datetime':
                row[col] = self._generate_datetime_value(col)
            elif self.column_types[col] == 'categorical':
                row[col] = self._generate_categorical_value(col)
            elif self.column_types[col] == 'text':
                row[col] = self._generate_text_value(col)

        return row

    def generate_dataframe(self, n_rows: int) -> pd.DataFrame:
        """
        Generate a DataFrame with n_rows of synthetic data.

        Parameters
        ----------
        n_rows : int
            The number of synthetic rows to generate.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing synthetic data.
        """
        rows = [self.generate_row() for _ in range(n_rows)]
        return pd.DataFrame(rows)