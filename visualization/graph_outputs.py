import json
import matplotlib.pyplot as plt

# root_directory = '../CIFAR10/resnet50_10ep'
root_directory = '../CIFAR10/simpleCNN_10ep'
metric_to_track = 'accuracy'
num_epochs = 10

metric_values = []

for i in range(num_epochs):
    filename = f'/metrics_{i}.json'
        
    # Read JSON data from file
    with open(root_directory + filename, 'r') as file:
        data = json.load(file)

    # Access the value
    metric_values.append(data[metric_to_track])
    # print(f"{metric_to_track}:", data[metric_to_track])

plt.plot(range(num_epochs), metric_values, marker='o', linestyle='-')
plt.title(f"{metric_to_track} over multiple epochs")
plt.xlabel('epoch')
plt.ylabel(f"{metric_to_track}")
plt.grid(True)
plt.show()