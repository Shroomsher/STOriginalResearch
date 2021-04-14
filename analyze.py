import os

def parce_input(input_la, opr):
    parce = input_la[1:]
    parce = parce[:-1]
    parce = parce.split(" ")
    parced_list = []
    for ex in range(len(parce)):
        parced_list.append(int(parce[ex]))
    apl = parced_list[int(opr)]
    return apl

directory = 'results/'

subfolders = [ f.path for f in os.scandir('results/') if f.is_dir() ]
for x in range(0,len(subfolders)):
    subfolders[x] += '/all_possible_results.csv'

import csv

correct_results = 0
incorrect_results = 0

for x in range(0, len(subfolders)):
    with open(subfolders[x], mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            if line_count == 1:
                OPR = row["one_position_rule"]
                OPR = int(OPR)-1
                OPR = str(OPR)
                line_count += 1
            if line_count > 1:
                input_array_string = row["input_array"]
                true_sight_outputs = row["true_sight_outputs"]
                true_sight_outputs = true_sight_outputs[:-2]
                parced = parce_input(input_array_string,OPR)
                print(parced)
                print(true_sight_outputs)
                if str(parced) == str(true_sight_outputs):
                    correct_results += 1
                    line_count += 1
                    print('Correct')
                else:
                    incorrect_results += 1
                    line_count += 1
                    print('Incorrect')

print("Correct results: " + str(correct_results))
print("Incorrect results: " + str(incorrect_results))