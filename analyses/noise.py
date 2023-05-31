import numpy as np
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sympy import *


class NoisyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.targets = [label for _, label in data]

    def __getitem__(self, index):
        return self.data[index]

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
    # scale_confusion_matrix = confusion_matrix

    # normalized the confusion matrix
    for i in range(len(confusion_matrix)):
        confusion_matrix[i] = confusion_matrix[i]/sum(confusion_matrix[i])

    # solve the scale factor to match the error_label_ratio
    scale_factor = []
    for original_data in client_datasets:
        scale_confusion_matrix = confusion_matrix
        label_distribution = {}
        for sample in original_data:
            if int(sample[1]) in label_distribution.keys():
                label_distribution[int(sample[1])] = label_distribution[int(sample[1])] + 1
            else:
                label_distribution[int(sample[1])] = 1
        base_confusion = 0
        for label in label_distribution.keys():
            base_confusion = base_confusion + scale_confusion_matrix[label][label] * label_distribution[label]
        x = symbols('x')
        z = solve(((base_confusion * x / len(original_data)) - (1 - error_label_ratio)), x)
        z = np.array(z).astype(float)
        scale_factor.append(z[0])
        for i in range(len(scale_confusion_matrix)):
            scale_confusion_matrix[i] = scale_confusion_matrix[i] * z[0]
            scale_confusion_matrix[i][i] = (scale_confusion_matrix[i][i] / z[0]) + (1 - scale_confusion_matrix[i][i] / z[0]) * (1 - z[0])
        original_labels = [sample[1] for sample in original_data]
        new_labels = [np.random.choice(class_num, p=scale_confusion_matrix[label]) for label in original_labels]
        new_dataset = [[original_data[i][0], new_labels[i]] for i in range(len(original_data))]

        noise_percentage = np.sum(np.array(original_labels) != np.array(new_labels)) / len(original_labels) * 100
        noise_percentages.append(noise_percentage)

        client_datasets_label_error.append(new_dataset)


    # for i in range(len(scale_confusion_matrix)):
    #     scale_confusion_matrix[i] = scale_confusion_matrix[i] * error_rate
    #     scale_confusion_matrix[i][i] = (scale_confusion_matrix[i][i] / error_rate) + (1 - scale_confusion_matrix[i][i] / error_rate) * (1 - error_rate)

    # for original_data in client_datasets:

    #     # Add label noise to dataset and calculate noise percentage
    #     original_labels = [sample[1] for sample in original_data]
    #     new_labels = [np.random.choice(class_num, p=scale_confusion_matrix[label]/sum(scale_confusion_matrix[label])) for label in original_labels]
    #     new_dataset = [[original_data[i][0], new_labels[i]] for i in range(len(original_data))]

    #     noise_percentage = np.sum(np.array(original_labels) != np.array(new_labels)) / len(original_labels) * 100
    #     noise_percentages.append(noise_percentage)

    #     client_datasets_label_error.append(new_dataset)

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

