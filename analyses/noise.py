import copy
import random

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = copy.deepcopy(data.targets)

    def __getitem__(self, index):
        return self.data[index][0], self.targets[index]

    def __len__(self):
        return len(self.data)


def inject_label_noise(client_datasets, class_num, error_ratio, error_var):
    """
    Add label noise to client datasets and log noise percentages to wandb.

    Args:
        client_datasets: a list of client datasets
        class_num: an integer indicating the number of classes.
        error_ratio: a float between 0 and 1 indicating the ratio of labels to be flipped.
        error_var: a float indicating the variance of the Gaussian distribution used to determine
            the level of label noise.

    Returns:
        A list of client datasets, and a list of noise percentages for each dataset
    """
    client_datasets_label_error = []
    noise_percentages = []
    for original_data in client_datasets:
        # Determine the level of label noise for this client dataset. The level is computed by normal distribution
        noisy_level = np.random.normal(error_ratio, error_var)
        noisy_level = max(noisy_level, 0)

        # Set the level of sparsity in the noise matrix.
        sparse_level = 0.4

        # Create a probability matrix for each label, where each element represents the probability of a label being assigned to that image.
        prob_matrix = np.full(class_num * class_num, 1 - noisy_level)

        # Set a random subset of elements in the probability matrix to zero to create sparsity.
        sparse_elements = np.random.choice(class_num * class_num, round(class_num * (class_num - 1) * sparse_level),
                                           replace=False)
        sparse_elements = sparse_elements[sparse_elements % (class_num + 1) != 0]
        prob_matrix[sparse_elements] = 0

        # Update prob_matrix
        prob_matrix = prob_matrix.reshape((class_num, class_num))
        for idx in range(len(prob_matrix)):
            non_zeros = np.count_nonzero(prob_matrix[idx])
            prob_element = 0 if non_zeros == 1 else (noisy_level) / (non_zeros - 1)
            prob_matrix[idx] = np.where(prob_matrix[idx] == 1 - noisy_level, prob_element, prob_matrix[idx])
            prob_matrix[idx, idx] = 1 - noisy_level

        # Add label noise to dataset and calculate noise percentage
        original_labels = [sample[1] for sample in original_data]
        new_labels = [np.random.choice(class_num, p=prob_matrix[label]) for label in original_labels]
        new_dataset = [[original_data[i][0], new_labels[i]] for i in range(len(original_data))]

        noise_percentage = np.sum(np.array(original_labels) != np.array(new_labels)) / len(original_labels) * 100
        noise_percentages.append(noise_percentage)

        client_datasets_label_error.append(new_dataset)

    return client_datasets_label_error, noise_percentages


def inject_label_noise_with_matrix(client_datasets, class_num, confusion_matrix, error_label_ratio):
    """
    Add label noise to client datasets and log noise percentages to wandb.

    Args:
        client_datasets: a list of client datasets
        class_num: an integer indicating the number of classes.
        confusion_matrix: the confusion matrix for the new labelling, which the size is class_num x class_num

    Returns:
        A list of client datasets, and a list of noise percentages for each dataset
    """
    client_datasets_label_error = []
    noise_percentages = []

    for original_data in client_datasets:
        new_dataset = original_data
        new_dataset = NoisyDataset(new_dataset)
        # new_dataset = [[original_data[i][0], original_data[i][1]] for i in range(len(new_dataset))]
        num_elements = len(original_data)
        num_elements_to_change = int(num_elements * error_label_ratio)
        # indices_to_change = random.sample(range(num_elements), num_elements_to_change)
        indices = random.sample(range(num_elements), num_elements)
        indices_to_change = []
        for index in indices:
            current_label_true = original_data[index][1]
            change_prob = confusion_matrix[current_label_true]
            if np.max(change_prob) < 0.80:
                indices_to_change.append(index)
            if len(indices_to_change) == num_elements_to_change:
                break

        changed_indices = set()
        for index in indices_to_change:
            current_label = original_data[index][1]
            new_label = np.random.choice(class_num,
                                         p=confusion_matrix[current_label] / sum(confusion_matrix[current_label]))
            while new_label == current_label or index in changed_indices:
                new_label = np.random.choice(class_num,
                                             p=confusion_matrix[current_label] / sum(confusion_matrix[current_label]))
            new_dataset.targets[index] = new_label
            changed_indices.add(index)

        original_labels = [sample[1] for sample in original_data]
        new_labels = [sample[1] for sample in new_dataset]
        noise_percentage = np.sum(np.array(original_labels) != np.array(new_labels)) / len(original_labels) * 100
        noise_percentages.append(noise_percentage)
        client_datasets_label_error.append(new_dataset)

    return client_datasets_label_error, noise_percentages

def plot_noise_percentage(original_datasets, noisy_datasets, run):
    """
    Function to calculate and plot label noise percentages for a list of datasets and upload it to wandb.

    Parameters:
    original_datasets (list): List of original PyTorch datasets.
    noisy_datasets (list): List of noisy PyTorch datasets.
    run (wandb.wandb_run.Run): The wandb run object to which the plot will be logged.

    Returns:
    None
    """
    # Compute label noise percentages
    label_noise_percentages = []

    for original_dataset, noisy_dataset in zip(original_datasets, noisy_datasets):
        original_labels = [label for _, label in original_dataset]
        noisy_labels = [label for _, label in noisy_dataset]

        # Compute noise percentage for this dataset
        noise_percentage = np.sum(np.array(original_labels) != np.array(noisy_labels)) / len(original_labels) * 100
        label_noise_percentages.append(noise_percentage)

    # Plot the label noise percentages as a histogram
    plt.hist(label_noise_percentages, bins=10, edgecolor='black')
    plt.title('Histogram of Label Noise Percentages')
    plt.xlabel('Label Noise Percentage')
    plt.ylabel('Count')

    # Save the plot to a file
    plt.savefig('label_noise_histogram.png')
    plt.close()  # Close the plot

    # Log the plot to wandb
    run.log({"label_noise_histogram": wandb.Image('label_noise_histogram.png')})
