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
import os

# --------------------------------------- VARIABLES ---------------------------------------

user_prompt = False
# MUST BE True or False

if user_prompt is False:

    training_iterations = random.randrange(10000, 1000000, 1000)
    # MUST BE AN INTEGER // Can be anywhere between 10000 and 1000000

    dataset_size = random.randrange(1000, 6000, 100)
    # MUST BE AN INTEGER // Can be anywhere between 1000-6000 // Should go down if the array_size goes up

    array_size = random.randrange(3, 14, 1)
    # MUST BE AN INTEGER // Can be anywhere between 3 and 12 // The larger it goes, the longer it will take to calculate

    one_position_rule = random.randrange(1, array_size, 1)
    # MUST BE AN INTEGER // Can be anywhere between 1 and array_size

    seed = None
    # MUST BE AN INTEGER OR 'None' // Can be anywhere between 0 and 1x10^32
else:
    training_iterations = int(
        input("Training iterations\nMUST BE AN INTEGER // Can be anywhere between 10000 and 1000000:\n"))
    dataset_size = int(input(
        "Dataset size\nMUST BE AN INTEGER // Can be anywhere between 1000-6000 // Should go down if the array_size goes up:\n"))
    array_size = int(input(
        "Array size\nMUST BE AN INTEGER // Can be anywhere between 3 and 25 // The larger it goes, the longer it will take to calculate:\n"))
    one_position_rule = int(
        input("One position rule\nMUST BE AN INTEGER // Can be anywhere between 1 and array_size:\n"))
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


def make_correct_outputs(input_list, one_position_rule_but_function):
    correct_outputs = np.zeros(len(input_list))
    for item in range(len(input_list)):
        if input_list[item, one_position_rule_but_function - 1] == 1:
            correct_outputs[item] = 1
            # The current rule: if there's a 1 in the first slot, the output should be 1

    return correct_outputs


def save_results(inputs, true_sight_outputs, calculated_outputs, spreadsheet_name):
    def recursive_rename(file_name_but_function, folder_path_but_function, f=0):
        new_file_path = os.path.join(folder_path_but_function, spreadsheet_name + "_" + str(f) + ".csv")
        file_path = os.path.join(folder_path_but_function, file_name)
        if os.path.exists(file_path):
            if os.path.exists(new_file_path):
                recursive_rename(file_name_but_function, folder_path_but_function, f=f + 1)
            else:
                os.rename(file_path, new_file_path)

    file_name = spreadsheet_name + ".csv"
    if not os.path.exists('./results/' + str(seed)):
        os.mkdir('./results/' + str(seed))
    folder_path = os.path.abspath('./results/' + str(seed) + '/')
    recursive_rename(file_name, folder_path)
    with open(os.path.join(folder_path, file_name), "w") as file:
        file.write(
            "input_array,true_sight_outputs,calculated_outputs,,synaptic_weight,training_iterations,dataset_size,array_size,one_position_rule,seed\n")
        parsed_weights = str(synaptic_weights).split('\n')
        parsed_weights = "".join(parsed_weights).strip()
        for kill in range(len(inputs)):
            parsed_inputs = str(inputs[kill]).split('.')
            parsed_inputs = "".join(parsed_inputs)
            file.write(f"{parsed_inputs},{true_sight_outputs[kill]},{calculated_outputs[kill]}")
            if kill == 0:
                file.write(
                    f",,{parsed_weights},{training_iterations},{dataset_size},{array_size},{one_position_rule},{seed}")
            file.write("\n")


# --------------------------------------- INITIALIZE DATASET & WEIGHTS ---------------------------------------

# Initialize and generate random dataset of 1s and 0s
random_array = np.random.rand(dataset_size * array_size)
rounded_array = np.around(random_array, decimals=0)
training_inputs = rounded_array.reshape(dataset_size, array_size)

# Correct outputs based off of dataset
training_outputs = make_correct_outputs(training_inputs, one_position_rule)

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

    if (iteration + 1) % 10000 == 0:
        print("Calculated iteration #" + str(iteration + 1))

print(f"Synaptic weights after training:\n{synaptic_weights}")

print("Average Error: " + str(np.average(error)))

# DEBUG:
# print(f"Outputs after training:\n{outputs}")
# print("Inputs: " + str(input_layer))

# --------------------------------------- SAVING RESULTS TO FILE ---------------------------------------

save_results(training_inputs, training_outputs, outputs, "training_results")

permutations = calculate_array_permutations(array_size)

save_results(permutations, make_correct_outputs(permutations, one_position_rule),
             calculate_outputs(permutations, synaptic_weights), "all_possible_results")

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



"""def main():

    # --------------------------------------- VARIABLES ---------------------------------------

    user_prompt = False
    # MUST BE True or False

    if user_prompt is False:

        training_iterations = random.randrange(10000,1000000,1000)
        # MUST BE AN INTEGER // Can be anywhere between 10000 and 1000000

        dataset_size = random.randrange(1000,6000,100)
        # MUST BE AN INTEGER // Can be anywhere between 1000-6000 // Should go down if the array_size goes up

        array_size = random.randrange(3,14,1)
        # MUST BE AN INTEGER // Can be anywhere between 3 and 12 // The larger it goes, the longer it will take to calculate

        one_position_rule = random.randrange(1,array_size,1)
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


    def make_correct_outputs(input_list, one_position_rule_but_function):
        correct_outputs = np.zeros(len(input_list))
        for item in range(len(input_list)):
            if input_list[item, one_position_rule_but_function - 1] == 1:
                correct_outputs[item] = 1
                # The current rule: if there's a 1 in the first slot, the output should be 1

        return correct_outputs


    def save_results(inputs, true_sight_outputs, calculated_outputs, spreadsheet_name):
        def recursive_rename(file_name_but_function, folder_path_but_function, f=0):
            new_file_path = os.path.join(folder_path_but_function, spreadsheet_name + "_" + str(f) + ".csv")
            file_path = os.path.join(folder_path_but_function, file_name)
            if os.path.exists(file_path):
                if os.path.exists(new_file_path):
                    recursive_rename(file_name_but_function, folder_path_but_function, f=f+1)
                else:
                    os.rename(file_path, new_file_path)

        file_name = spreadsheet_name + ".csv"
        if not os.path.exists('./results/' + str(seed)):
            os.mkdir('./results/' + str(seed))
        folder_path = os.path.abspath('./results/' + str(seed) + '/')
        recursive_rename(file_name, folder_path)
        with open(os.path.join(folder_path, file_name), "w") as file:
            file.write("input_array,true_sight_outputs,calculated_outputs,,synaptic_weight,training_iterations,dataset_size,array_size,one_position_rule,seed\n")
            parsed_weights = str(synaptic_weights).split('\n')
            parsed_weights = "".join(parsed_weights).strip()
            for kill in range(len(inputs)):
                parsed_inputs = str(inputs[kill]).split('.')
                parsed_inputs = "".join(parsed_inputs)
                file.write(f"{parsed_inputs},{true_sight_outputs[kill]},{calculated_outputs[kill]}")
                if kill == 0:
                    file.write(f",,{parsed_weights},{training_iterations},{dataset_size},{array_size},{one_position_rule},{seed}")
                file.write("\n")


    # --------------------------------------- INITIALIZE DATASET & WEIGHTS ---------------------------------------

    # Initialize and generate random dataset of 1s and 0s
    random_array = np.random.rand(dataset_size * array_size)
    rounded_array = np.around(random_array, decimals=0)
    training_inputs = rounded_array.reshape(dataset_size, array_size)


    # Correct outputs based off of dataset
    training_outputs = make_correct_outputs(training_inputs, one_position_rule)

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

        if (iteration + 1) % 10000 == 0:
            print("Calculated iteration #" + str(iteration+1))

    print(f"Synaptic weights after training:\n{synaptic_weights}")

    print("Average Error: " + str(np.average(error)))

    # DEBUG:
    # print(f"Outputs after training:\n{outputs}")
    # print("Inputs: " + str(input_layer))

    # --------------------------------------- SAVING RESULTS TO FILE ---------------------------------------
    
    save_results(training_inputs, training_outputs, outputs, "training_results")

    permutations = calculate_array_permutations(array_size)

    save_results(permutations, make_correct_outputs(permutations, one_position_rule), calculate_outputs(permutations, synaptic_weights), "all_possible_results")


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

if __name__ == '__main__':
    for yeehaw in range(34):
        main()"""
