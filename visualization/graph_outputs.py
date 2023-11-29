import json
import matplotlib.pyplot as plt

root_directories = ['../CIFAR10/resnet50_10ep', '../CIFAR10/simpleCNN_10ep']
model_names = ['resnet50', 'simpleCNN']
metric_to_track = 'accuracy'

num_epochs = 10
metric_values = []

# Load metrics for each model
for root_directory in root_directories:
    cur_metric_values = []

    for i in range(num_epochs):
        filename = f'/metrics_{i}.json'
            
        # Read JSON data from file
        with open(root_directory + filename, 'r') as file:
            data = json.load(file)

        # Access the value
        cur_metric_values.append(data[metric_to_track])
    metric_values.append(cur_metric_values)

for i, metric_values in enumerate(metric_values):
    plt.plot(range(num_epochs), metric_values, label=model_names[i], marker='o', linestyle='-')

plt.title(f"{metric_to_track} over multiple epochs")
plt.xlabel('epoch')
plt.ylabel(f"{metric_to_track}")
plt.grid(True)
plt.legend(loc='lower right') 
plt.show()