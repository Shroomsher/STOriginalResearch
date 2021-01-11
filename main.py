# --------------------------------------- EXPLANATION ---------------------------------------
# This is the near final product of my ST OR.
# In it's current state (v1.5), it can:
# Generate its own dataset and calculate outputs.
# Train it's self on that generated dataset.
# Predict accurately ___% of the time what the output will be using a sigmoid function.
# Made, tested, fixed, and polished in roughly 45 hours (in the current state of v1.5).

# --------------------------------------- MODULES ---------------------------------------

import numpy as np
import random


# --------------------------------------- FUNCTIONS ---------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def return_one_or_zero(x):
    if int(x) >= .5:
        x = 1
    else:
        x = 0

    print("Rounding to " + str(x) + ".")

    return x


# --------------------------------------- VARIABLES ---------------------------------------

number_of_layers = 100000

dataset_size = 1000

fixed_seed = None

random_seed = random.randint(0, 1000000000)


# Print initialized variables
print("Number of layers: " + str(number_of_layers))
print("Dataset size: " + str(dataset_size))


# Set random or fixed seed
if isinstance(fixed_seed, int) or isinstance(fixed_seed, float):
    np.random.seed(fixed_seed)
    print("Starting seed: " + str(fixed_seed))
else:
    np.random.seed(random_seed)
    print("Starting seed: " + str(random_seed))

# --------------------------------------- INITIALIZE EVERYTHING ---------------------------------------

# Initialize output array
training_outputs = np.random.rand(dataset_size)

# Initialize and generate random dataset of 1s and 0s
random_array = np.random.rand(dataset_size * 3)
rounded_array = np.around(random_array, decimals=0)
reshaped_array = rounded_array.reshape(dataset_size, 3)
training_inputs = reshaped_array

# Correct outputs based off of dataset
for iteration in range(dataset_size):
    if reshaped_array[iteration, 0] == 1:
        training_outputs[iteration] = 1
    else:
        training_outputs[iteration] = 0

# Initialize synaptic weights
synaptic_weights = 2 * np.random.random(3) - 1

print(f"Random start synaptic weights:\n{synaptic_weights}")

# --------------------------------------- TRAINING ---------------------------------------
# Train the algorithm on the generated dataset
for iteration in range(number_of_layers):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # Calculate error
    error = training_outputs - outputs
    # Calculate adjustments to be made to the weights
    adjustments = error * sigmoid_derivative(outputs)
    # Make adjustments to the weights
    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)

    if (iteration + 1) % 1000 == 0:
        print("Calculated iteration #" + str(iteration+1))

print(f"Synaptic weights after training:\n{synaptic_weights}")

# DEBUG:
# print(f"Outputs after training:\n{outputs}")
# print("Inputs: " + str(input_layer))

print("Average Error: " + str(np.average(error)))

# --------------------------------------- USER INPUT ---------------------------------------

user_array = np.random.random(3)

for i in range(3):
    user_array_input = input("AC" + str(i + 1) + ": ")

    if isinstance(user_array_input, int) or isinstance(user_array_input, float):
        user_array_input = random.randint(0, 1)
        print("Non-int detected, defaulting to " + str(user_array_input) + ".")
    else:
        user_array_input = return_one_or_zero(user_array_input)

    user_array[i] = user_array_input

user_output = sigmoid(np.dot(user_array, synaptic_weights))
rounded_user_output = np.around(user_output, decimals=0)

print(f"Algorithm prediction, based on the training on the random seed \'" + str(random_seed) + f"\'")
print(f"Raw output:\n{user_output}")
print(f"Rounded output:\n{rounded_user_output}")
