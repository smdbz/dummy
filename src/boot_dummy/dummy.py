"""
Generate synthetic data with realistic statistical properties
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Any, Optional

class GenerateData:
    """
    A class for generating synthetic data and analyzing datasets.
    
    This class provides functionality to generate synthetic data, analyze existing 
    datasets, and create new data that matches statistical properties of real data.
    """

    def __init__(self, num_records: int = 1000) -> None:
        """Initialize with the specified number of records.
        
        Args:
            num_records: Number of records to generate (default: 1000)
            
        Raises:
            ValueError: If num_records is not a positive integer
        """
        if not isinstance(num_records, int) or num_records <= 0:
            raise ValueError("num_records must be a positive integer")
        self.num_records = num_records

    def generate(self) -> pd.DataFrame:
        """Generate synthetic customer data using the instance's num_records.
        
        Returns:
            DataFrame with Age, Income, CreditScore, and LoanAmount columns
        """
        return self.generate_customer_data(self.num_records)

    @staticmethod
    def generate_customer_data(num_records: int) -> pd.DataFrame:
        """Generate synthetic customer data with predefined rules.
        
        Creates data with Age, Income, CreditScore, and LoanAmount columns.
        
        Args:
            num_records: Number of customer records to generate
            
        Returns:
            DataFrame containing synthetic customer data
        """
        data = []
        for _ in range(num_records):
            age = np.random.randint(18, 80)

            # Rule: Income is loosely based on age
            base_income = 20000 + (age - 18) * 1000
            income = np.random.normal(base_income, base_income * 0.2)

            # Rule: Credit score is influenced by age and income
            credit_score = min(850, max(300, int(600 + (age / 80) * 100 + (income / 100000) * 100 + np.random.normal(0,
                                                                                                                     50))))

            # Rule: Loan amount is based on income and credit score
            max_loan = income * (credit_score / 600)
            loan_amount = np.random.uniform(0, max_loan)

            data.append([age, income, credit_score, loan_amount])

        return pd.DataFrame(data, columns=['Age', 'Income', 'CreditScore', 'LoanAmount'])

    @staticmethod
    def enhanced_describe(df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced descriptive statistics for a DataFrame.
        
        Extends pandas' describe() with additional statistics like skewness,
        kurtosis, and percentiles for numeric columns, and value counts for
        categorical columns.
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            DataFrame containing detailed statistics for each column
            
        Raises:
            ValueError: If input is not a pandas DataFrame or is empty
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        stats = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Get basic statistics and add additional ones
                basic_stats = df[column].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
                
                # Add more advanced statistics
                stats[column] = {
                    **basic_stats,
                    'skewness': df[column].skew(),
                    'kurtosis': df[column].kurtosis(),
                    'missing_values': df[column].isnull().sum(),
                    'missing_percent': (df[column].isnull().sum() / len(df)) * 100,
                    'unique_values': df[column].nunique(),
                    'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                    'variance': df[column].var()
                }
            else:
                # For non-numeric columns
                stats[column] = {
                    'count': df[column].count(),
                    'unique_values': df[column].nunique(),
                    'missing_values': df[column].isnull().sum(),
                    'missing_percent': (df[column].isnull().sum() / len(df)) * 100,
                    'mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                    'value_counts': df[column].value_counts().head().to_dict()
                }

        return pd.DataFrame(stats)
        
    @staticmethod
    def generate_from_stats(stats_df: pd.DataFrame, num_records: int = 1000) -> pd.DataFrame:
        """Generate synthetic data based on statistics from enhanced_describe.
        
        Creates data that matches statistical properties of an original dataset.
        Handles both numeric and categorical columns, preserving distributions.
        
        Args:
            stats_df: Statistics DataFrame from enhanced_describe()
            num_records: Number of records to generate (default: 1000)
            
        Returns:
            Synthetic data matching the statistical properties of the original data
            
        Raises:
            ValueError: If stats_df is not a pandas DataFrame
            ValueError: If num_records is not a positive integer
        """
        if not isinstance(stats_df, pd.DataFrame):
            raise ValueError("stats_df must be a pandas DataFrame")
        
        if not isinstance(num_records, int) or num_records <= 0:
            raise ValueError("num_records must be a positive integer")
            
        # Initialize empty DataFrame to store generated data
        synthetic_data = pd.DataFrame(index=range(num_records))
        
        # Generate data for each column based on its statistics
        for column in stats_df.columns:
            col_stats = stats_df[column]
            
            # Check if this is a numeric column (has 'mean' and 'std' statistics)
            if 'mean' in col_stats and 'std' in col_stats:
                # Generate numeric data
                if col_stats['std'] > 0:
                    # Use normal distribution with truncation for most numeric columns
                    data = np.random.normal(
                        loc=col_stats['mean'],
                        scale=col_stats['std'],
                        size=num_records
                    )
                    
                    # Truncate to min/max if available
                    if 'min' in col_stats and 'max' in col_stats:
                        data = np.clip(data, col_stats['min'], col_stats['max'])
                    
                    # If skewness is significant, adjust the distribution
                    if 'skewness' in col_stats and abs(col_stats['skewness']) > 1:
                        # For positive skew, use lognormal-like adjustment
                        if col_stats['skewness'] > 1:
                            # Shift data to be positive
                            min_val = data.min()
                            if min_val < 0:
                                data = data - min_val + 1
                            
                            # Apply power transformation based on skewness
                            skew_factor = min(col_stats['skewness'] / 2, 3)  # Limit the factor
                            data = np.power(data, skew_factor)
                            
                            # Rescale to match original mean and std
                            data = (data - data.mean()) / data.std() * col_stats['std'] + col_stats['mean']
                            
                            # Clip to original bounds
                            if 'min' in col_stats and 'max' in col_stats:
                                data = np.clip(data, col_stats['min'], col_stats['max'])
                else:
                    # If std is 0, use constant value
                    data = np.full(num_records, col_stats['mean'])
                
                synthetic_data[column] = data
                
                # If the original data was integer type (based on column name or other heuristics)
                if column in ['Age', 'CreditScore'] or ('unique_values' in col_stats and col_stats['unique_values'] < 100):
                    synthetic_data[column] = synthetic_data[column].round().astype(int)
            
            # Handle categorical columns (those with value_counts)
            elif 'value_counts' in col_stats and col_stats['value_counts']:
                categories = list(col_stats['value_counts'].keys())
                probabilities = list(col_stats['value_counts'].values())
                # Normalize probabilities
                probabilities = [p / sum(probabilities) for p in probabilities]
                
                # Generate categorical data
                synthetic_data[column] = np.random.choice(
                    categories,
                    size=num_records,
                    p=probabilities
                )
        
        return synthetic_data
        
    @staticmethod
    def load_dataset(file_path: str, **kwargs) -> pd.DataFrame:
        """Load a dataset from a file (CSV, Excel, JSON, Parquet).
        
        Automatically detects file format based on extension.
        
        Args:
            file_path: Path to the dataset file
            **kwargs: Additional arguments for the pandas read function
            
        Returns:
            Loaded dataset as a pandas DataFrame
            
        Raises:
            ValueError: If the file format is unsupported or file doesn't exist
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path, **kwargs)
        elif file_ext == '.json':
            return pd.read_json(file_path, **kwargs)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    @classmethod
    def generate_from_dataset(cls, 
                             dataset_path: Optional[str] = None, 
                             dataset: Optional[pd.DataFrame] = None,
                             num_records: int = 1000,
                             **kwargs) -> Dict[str, Any]:
        """Complete workflow to load a dataset, analyze it, and generate synthetic data.
        
        Combines loading data, generating statistics, and creating synthetic data
        in a single operation.
        
        Args:
            dataset_path: Path to the dataset file (optional)
            dataset: Dataset as DataFrame (optional, alternative to dataset_path)
            num_records: Number of synthetic records to generate (default: 1000)
            **kwargs: Additional arguments for load_dataset method
            
        Returns:
            Dictionary with keys:
                - 'original_data': The original dataset
                - 'stats': The statistics from enhanced_describe
                - 'synthetic_data': The generated synthetic data
                
        Raises:
            ValueError: If neither dataset_path nor dataset is provided
            ValueError: If dataset is provided but is not a pandas DataFrame
        """
        # Load the dataset
        if dataset is not None:
            if not isinstance(dataset, pd.DataFrame):
                raise ValueError("dataset must be a pandas DataFrame")
            data = dataset
        elif dataset_path is not None:
            data = cls.load_dataset(dataset_path, **kwargs)
        else:
            raise ValueError("Either dataset_path or dataset must be provided")
            
        # Generate statistics
        stats = cls.enhanced_describe(data)
        
        # Generate synthetic data
        synthetic_data = cls.generate_from_stats(stats, num_records)
        
        # Return all components
        return {
            'original_data': data,
            'stats': stats,
            'synthetic_data': synthetic_data
        }
