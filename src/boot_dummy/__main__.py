from .dummy import GenerateData

def main():
    # Generate sample data with 10 records
    data = GenerateData.generate_customer_data(1000)
    print("Generated customer data:")
    print(data)

if __name__ == "__main__":
    main()
