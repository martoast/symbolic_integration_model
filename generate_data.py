import sympy as sp
import random
import pandas as pd

# Define the symbol
x = sp.symbols('x')

# Function to generate random polynomials
def random_polynomial(degree=3):
    return sum(random.randint(-10, 10) * x**i for i in range(degree + 1))

# Function to generate random trigonometric functions
def random_trigonometric():
    functions = [sp.sin(x), sp.cos(x), sp.tan(x)]
    return random.choice(functions)

# Function to generate random exponential and logarithmic functions
def random_exponential_logarithmic():
    functions = [sp.exp(x), sp.log(x)]
    return random.choice(functions)

# Function to combine different types of functions
def random_combination():
    funcs = [
        random_polynomial(),
        random_trigonometric(),
        random_exponential_logarithmic()
    ]
    combination = random.choice(funcs)
    if random.random() > 0.5:
        combination += random.choice(funcs)
    if random.random() > 0.5:
        combination *= random.choice(funcs)
    return combination

# Generate a list of random functions
def generate_random_functions(n=10):
    functions = []
    for _ in range(n):
        funcs = [
            random_polynomial(),
            random_trigonometric(),
            random_exponential_logarithmic(),
            random_combination()
        ]
        functions.append(random.choice(funcs))
    return functions

# Generate the data and save to a CSV file
def generate_and_save_data(n=100, filename='data.csv'):
    random_functions = generate_random_functions(n)
    data = []
    for func in random_functions:
        derivative = sp.diff(func, x)
        data.append((str(derivative), str(func)))

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['Derivative', 'Original_Function'])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Generate and save training data
generate_and_save_data(1000, 'training_data.csv')

# Generate and save validation data
generate_and_save_data(200, 'validation_data.csv')  # Adjust the number of samples as needed