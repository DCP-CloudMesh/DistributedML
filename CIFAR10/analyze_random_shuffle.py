"""
This script is used to analyze the results of the random shuffle distributed experiments.

random_shuffle/
    - experiment_{i}/ for i in range(10)
        - output/
            - metrics_final.json
            - model.pth
random_shuffle_distributed/
    - experiment_{i}/ for i in range(10)
        - output/
            - metrics_final.json
            - model.pth

I want to load the metrics_final.json files and load the 'loss', 'accuracy', 'precision', 'recall', 'f1' values.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics, metrics_distributed, metric_name, title, y_label, save_path):
    _, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(10)
    ax.scatter(x, metrics[metric_name], label="Random Shuffle")
    ax.scatter(x, metrics_distributed[metric_name], label="Random Shuffle Distributed")
    ax.set_xlabel("Experiment Number")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.savefig(save_path)


# i want a candle graph with two values in the x axis, the left one being the metric
# and the right one being the metric_distributed. I want to plot a candle with a black line denoting
# the mean, the box denoting the q1 and q3, and the whiskers denoting the min and max.
def plot_candle_graph_with_mean_and_std(
    metrics, metrics_distributed, metric_name, title, y_label, save_path
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare the data in the order: [metrics, metrics_distributed]
    data = [metrics[metric_name], metrics_distributed[metric_name]]

    # Create the box-and-whisker plot
    # whis=[0, 100] extends the whiskers to the min and max values
    # showmeans=True draws a line for the mean, and meanline=True ensures it's a solid line
    box = ax.boxplot(
        data,
        whis=[0, 100],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        labels=["Random Shuffle", "Random Shuffle Distributed"],
        boxprops=dict(facecolor="white", edgecolor="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        meanprops=dict(color="black", linewidth=2, linestyle="-"),
        medianprops=dict(color="none"),  # Hide the standard median line
    )

    ax.set_title(title)
    ax.set_ylabel(y_label)
    plt.savefig(save_path)
    plt.close()


def main():
    # load the metrics
    metrics_rs = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    metrics_rsd = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    for i in range(10):
        ind = i + 1
        with open(
            f"CIFAR10/random_shuffle/experiment_{ind}/output/metrics_final.json", "r"
        ) as f:
            # metrics_rs.append(json.load(f))
            values = json.load(f)
            metrics_rs["loss"].append(values["loss"])
            metrics_rs["accuracy"].append(values["accuracy"])
            metrics_rs["precision"].append(values["precision"])
            metrics_rs["recall"].append(values["recall"])
            metrics_rs["f1"].append(values["f1"])
        with open(
            f"CIFAR10/random_shuffle_distributed/experiment_{ind}/output/metrics_final.json",
            "r",
        ) as f:
            # metrics_rsd.append(json.load(f))
            values = json.load(f)
            metrics_rsd["loss"].append(values["loss"])
            metrics_rsd["accuracy"].append(values["accuracy"])
            metrics_rsd["precision"].append(values["precision"])
            metrics_rsd["recall"].append(values["recall"])
            metrics_rsd["f1"].append(values["f1"])

    data_path = "rs_metrics"

    plot_metrics(
        metrics_rs,
        metrics_rsd,
        "loss",
        "Loss over 10 experiments",
        "Loss",
        f"{data_path}/loss.png",
    )
    plot_metrics(
        metrics_rs,
        metrics_rsd,
        "accuracy",
        "Accuracy over 10 experiments",
        "Accuracy",
        f"{data_path}/accuracy.png",
    )
    plot_candle_graph_with_mean_and_std(
        metrics_rs,
        metrics_rsd,
        "accuracy",
        "Accuracy over 10 experiments",
        "Accuracy",
        f"{data_path}/accuracy_candle.png",
    )
    # plot_metrics(
    #     metrics_rs,
    #     metrics_rsd,
    #     "precision",
    #     "Precision over 10 experiments",
    #     "Precision",
    #     "precision.png",
    # )
    # plot_metrics(
    #     metrics_rs,
    #     metrics_rsd,
    #     "recall",
    #     "Recall over 10 experiments",
    #     "Recall",
    #     "recall.png",
    # )
    plot_metrics(
        metrics_rs,
        metrics_rsd,
        "f1",
        "F1 over 10 experiments",
        "F1",
        f"{data_path}/f1.png",
    )
    plot_candle_graph_with_mean_and_std(
        metrics_rs,
        metrics_rsd,
        "f1",
        "F1 over 10 experiments",
        "F1",
        f"{data_path}/f1_candle.png",
    )


if __name__ == "__main__":
    main()
