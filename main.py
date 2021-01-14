# --------------------------------------- PREFACE ---------------------------------------
# This is the near final product of my ST OR.
# In it's current state (v1.8), it can:
# Generate its own dataset and calculate outputs.
# Train it's self on that generated dataset.
# Predict accurately ___% of the time what the output will be using a sigmoid function.
# Made, tested, fixed, and polished in roughly 53 hours (in the current state of v1.8).

# As discovered later, the math here isn't advanced enough to recognize and understand if two variables are required for
# the output to be 1, so applying any outside datasets and running them through will not work. If I had the time, I'd
# look for another function that could get more complex with multi-variable weighted output, but I only have 3 days
# left... oh boy!

# --------------------------------------- MODULES ---------------------------------------

import numpy as np
import random
import itertools

# --------------------------------------- VARIABLES ---------------------------------------

user_prompt = False
# MUST BE True or False

if user_prompt is False:

    training_iterations = 500000
    # MUST BE AN INTEGER // Can be anywhere between 10000 and 1000000

    dataset_size = 6000
    # MUST BE AN INTEGER // Can be anywhere between 1000-6000 // Should go down if the array_size goes up

    array_size = 10
    # MUST BE AN INTEGER // Can be anywhere between 3 and 25 // The larger it goes, the longer it will take to calculate

    one_position_rule = 6
    # MUST BE AN INTEGER // Can be anywhere between 1 and array_size

    seed = None
    # MUST BE AN INTEGER OR 'None' // Can be anywhere between 0 and 1x10^32
else:
    training_iterations = int(input("Training iterations\nMUST BE AN INTEGER // Can be anywhere between 10000 and 1000000:\n"))
    dataset_size = int(input("Dataset size\nMUST BE AN INTEGER // Can be anywhere between 1000-6000 // Should go down if the array_size goes up:\n"))
    array_size = int(input("Array size\nMUST BE AN INTEGER // Can be anywhere between 3 and 25 // The larger it goes, the longer it will take to calculate:\n"))
    one_position_rule = int(input("One position rule\nMUST BE AN INTEGER // Can be anywhere between 1 and array_size:\n"))
    seed = input("Seed\nMUST BE AN INTEGER OR 'None' // Can be anywhere between 0 and 1x10^32:\n")
    seed = seed.lower()
    if seed != 'none':
        seed = int(seed)
    else:
        seed = None

# Print initialized variables
print("Number of layers: " + str(training_iterations))
print("Dataset size: " + str(dataset_size))
print("Array size: " + str(array_size))
print("Rule: The '1' should be in slot #" + str(one_position_rule))


# Set random seed at either the given value or a random number
is_set_seed = True
if seed is None:
    is_set_seed = False
    seed = random.randint(0, 1000000000)

np.random.seed(seed)
print("Starting seed: " + str(seed))


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


def calculate_outputs(x, weights):
    return sigmoid(np.dot(x, weights))


def calculate_array_permutations(array_size_but_function):
    return np.array(list(itertools.product([0, 1], repeat=array_size_but_function)))


def run_through_permutations(permutations, weights):
    input_output = calculate_outputs(permutations, weights)




# --------------------------------------- INITIALIZE EVERYTHING ---------------------------------------

# Initialize output array
training_outputs = np.random.rand(dataset_size)

# Initialize and generate random dataset of 1s and 0s
random_array = np.random.rand(dataset_size * array_size)
rounded_array = np.around(random_array, decimals=0)
reshaped_array = rounded_array.reshape(dataset_size, array_size)
training_inputs = reshaped_array

# Correct outputs based off of dataset
for iteration in range(dataset_size):
    if reshaped_array[iteration, one_position_rule-1] == 1:
        training_outputs[iteration] = 1
        # The current rule: if there's a 1 in the first slot, the output should be 1
    else:
        training_outputs[iteration] = 0

# Initialize synaptic weights
synaptic_weights = 2 * np.random.random(array_size) - 1

print(f"Random start synaptic weights:\n{synaptic_weights}")

# --------------------------------------- TRAINING ---------------------------------------
# Train the algorithm on the generated dataset
for iteration in range(training_iterations):

    input_layer = training_inputs
    outputs = calculate_outputs(input_layer, synaptic_weights)
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
    user_array = np.random.random(array_size)

    # Ask for and copy each of the inputs to the array
    for i in range(array_size):
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
        print("Confidence level:\n~" + str(user_output*100) + "%")
    else:
        print("Confidence level:\n~" + str(100-user_output * 100) + "%")

    # DEBUG
    # print("Raw user output:\n" + str(user_output))

    kill_loop = input("Calculate again? (y/n):\n")
    continuing = kill_loop
