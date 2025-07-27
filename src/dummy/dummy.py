"""
Generate dummy data based on a list of requirements
"""

import numpy as np
import pandas as pd

class GenerateData:
    """
    Takes arguments as input and generates a table based on the arguments.
    """
    def __init__(self, num_records):
        self.num_records = num_records


    @classmethod
    def from_dict(cls):
        """
        Allows the creation of a new class from a provided dictionary
        :return:
        """

    @staticmethod
    def generate_customer_data(num_records):
        """
        MVP, just make a dataset.
        Code stolen from https://www.datacamp.com/tutorial/synthetic-data-generation
        :param num_records:
        :return:
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
