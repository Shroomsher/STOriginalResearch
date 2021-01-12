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

training_iterations = 100000

dataset_size = 5000

seed = None
# 48071936


# Print initialized variables
print("Number of layers: " + str(training_iterations))
print("Dataset size: " + str(dataset_size))


# Set random seed at either the given value or a random number
is_set_seed = True
if seed is None:
    is_set_seed = False
    seed = random.randint(0, 1000000000)

np.random.seed(seed)
print("Starting seed: " + str(seed))

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
    if reshaped_array[iteration, 0] == 1 and reshaped_array[iteration, 2]:
        training_outputs[iteration] = 1
        # The current rule: if 1s are in both the 0 and 2 position of the array, the output should be 1
    else:
        training_outputs[iteration] = 0

# Initialize synaptic weights
synaptic_weights = 2 * np.random.random(3) - 1

print(f"Random start synaptic weights:\n{synaptic_weights}")

# --------------------------------------- TRAINING ---------------------------------------
# Train the algorithm on the generated dataset
for iteration in range(training_iterations):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    # Calculate error
    error = training_outputs - outputs
    # Calculate adjustments to be made to the weights
    adjustments = error * sigmoid_derivative(outputs)
    # Make adjustments to the weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

    if (iteration + 1) % 1000 == 0:
        print("Calculated iteration #" + str(iteration+1))

print(f"Synaptic weights after training:\n{synaptic_weights}")

print("Average Error: " + str(np.average(error)))

# DEBUG:
# print(f"Outputs after training:\n{outputs}")
# print("Inputs: " + str(input_layer))

# --------------------------------------- USER INPUT ---------------------------------------

continuing = 'y'

# Start a loop
while continuing != 'n':

    # Initialize the user's input array
    user_array = np.random.random(3)

    # Ask for and copy each of the inputs to the array
    for i in range(3):
        user_array_input = input("AC" + str(i + 1) + ": ")
        user_array_input = int(user_array_input)
        user_array_input = return_one_or_zero(user_array_input)

        # if isinstance(user_array_input, int) is True or isinstance(user_array_input, float) is True:
        #     user_array_input = return_one_or_zero(user_array_input)
        # else:
        #     user_array_input = random.randint(0, 1)
        #     print("Non-int detected, defaulting to " + str(user_array_input) + ".")

        user_array[i] = user_array_input

    # Calculate outputs
    user_output = sigmoid(np.dot(user_array, synaptic_weights))
    rounded_user_output = np.around(user_output, decimals=0)

    # Determine whether fixed or random seed, and print respectfully
    if is_set_seed:
        print(f"Algorithm prediction, based on the training on the fixed seed \'" + str(seed) + f"\'")
    else:
        print(f"Algorithm prediction, based on the training on the random seed \'" + str(seed) + f"\'")

    print("Educated guess on output:\n" + str(rounded_user_output))

    # Calculate confidence level
    if rounded_user_output == 1:
        print("Confidence level:\n~" + str(np.around(user_output*100, decimals=3)) + "%")
    else:
        print("Confidence level:\n~" + str(np.around(100-user_output * 100, decimals=3)) + "%")

    # DEBUG
    # print("Raw user output:\n" + str(user_output))

    kill_loop = input("Calculate again? (y/n):\n")
    continuing = kill_loop
