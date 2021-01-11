import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


random_seed = 1
np.random.seed(random_seed)

training_outputs = np.random.rand(1000)

random_array = np.random.rand(3000)
rounded_array = np.around(random_array, decimals=0)
reshaped_array = rounded_array.reshape(1000, 3)
training_inputs = reshaped_array

for iteration in range(1000):
    if reshaped_array[iteration, 0] == 1:
        training_outputs[iteration] = 1
    else:
        training_outputs[iteration] = 0

synaptic_weights = 2 * np.random.random(3) - 1

print(f"Random start synaptic weights:\n{synaptic_weights}")

for iteration in range(10000):

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights = synaptic_weights + np.dot(input_layer.T, adjustments)

    print("Calculating iteration #" + str(iteration+1))

print(f"Synaptic weights after training:\n{synaptic_weights}")

# print(f"Outputs after training:\n{outputs}")

# print("Inputs: " + str(input_layer))

print("Average Error: " + str(np.average(error)))

user_array_var1 = input("Array-cast 1: ")
user_array_var2 = input("Array-cast 2: ")
user_array_var3 = input("Array-cast 3: ")

user_array = np.random.random(3)
user_array[0] = user_array_var1
user_array[1] = user_array_var2
user_array[2] = user_array_var3

user_output = sigmoid(np.dot(user_array, synaptic_weights))
rounded_user_output = np.around(user_output, decimals=0)

print(f"Algorithm prediction, based on the training on the random seed \'" + str(random_seed) + f"\'")
print(f"Raw output:\n{user_output}")
print(f"Rounded output:\n{rounded_user_output}")
