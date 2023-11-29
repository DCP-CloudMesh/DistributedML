import json
import matplotlib.pyplot as plt

# ----------------------------------------
# Initialization
# ----------------------------------------
root_directories = ['../CIFAR10/simpleCNNdistributed_10ep_002', 
                    '../CIFAR10/simpleCNN_10ep_002',
                    '../CIFAR10/resnet18distributed_10ep_001']
model_names = ['simpleCNN distributed', 'simpleCNN', 'resnet18 distributed']
metric_to_track = 'loss'

num_epochs = 10
metric_values = []

# ----------------------------------------
# Loading Metrics
# ----------------------------------------
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

# ----------------------------------------
# Plotting 
# ----------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed

for i, metric_values in enumerate(metric_values):
    ax.plot(range(num_epochs), metric_values, label=f'Model {i+1}', marker='o', linestyle='-')

ax.set_title(f"{metric_to_track} over multiple epochs")
ax.set_xlabel('epoch')
ax.set_ylabel(f"{metric_to_track}")
ax.grid(True)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()  
plt.show()