import numpy as np
import torch
from tqdm import tqdm
import ray

from aggregators.scaffold import Scaffold

def distributed_scaffold(aggregator,
                         client_trainers,
                         client_dataset_refs,
                         client_num_per_round,
                         global_model,
                         round_idx,
                         scheduler,
                         device,
                         precision):
    # Select random clients for each round
    sampled_clients_idx = np.random.choice(len(client_dataset_refs), client_num_per_round, replace=False)
    print(f"selected clients: {sampled_clients_idx}")

    # Initialize lists to store updates, weights, local metrics, and control variates
    all_updates, all_weights, all_local_metrics = [], [], []
    all_control_variates = []

    # Initialize control variates
    global_control_variates = initialize_control_variates(global_model)

    # Iterate over the sampled clients in chunks equal to the number of client trainers
    for i in tqdm(range(0, len(sampled_clients_idx), len(client_trainers))):
        # Initialize list to store remote steps
        remote_steps = []

        # Iterate over the client trainers
        for j, client_trainer in enumerate(client_trainers):
            idx = i + j
            if idx >= len(sampled_clients_idx):
                break

            # Update the remote client_trainer with the latest global model and scheduler state
            client_trainer.update.remote(global_model.state_dict(), scheduler.state_dict())

            # Perform a remote training step on the client_trainer
            if precision != 'float32':
                remote_step = client_trainer.step_low_precision.remote(sampled_clients_idx[idx],
                                                                       client_dataset_refs[sampled_clients_idx[idx]],
                                                                       round_idx,
                                                                       precision,
                                                                       device=device,
                                                                       global_control_variates=global_control_variates)
            else:
                remote_step = client_trainer.step.remote(sampled_clients_idx[idx],
                                                         client_dataset_refs[sampled_clients_idx[idx]],
                                                         round_idx,
                                                         device=device,
                                                         global_control_variates=global_control_variates)
            remote_steps.append(remote_step)

        # Retrieve remote steps results
        print(f"length of steps: {len(remote_steps)}")
        results = ray.get(remote_steps)

        # Separate updates, control variates, and other metrics
        for result in results:
            update, control_variate, num_client_samples, local_metrics = result
            all_updates.append(update)
            all_control_variates.append(control_variate)
            all_weights.append(num_client_samples)
            all_local_metrics.append(local_metrics)

        torch.cuda.empty_cache()

    # Calculate the average local metrics
    local_metrics_avg = {key: sum(metric[key] for metric in all_local_metrics if key in metric) / len(all_local_metrics)
                         for key in all_local_metrics[0]}
    print(all_local_metrics)

    # Update the global model using the aggregator
    state_n = aggregator.step(all_updates, all_control_variates, all_weights, round_idx)
    global_model.load_state_dict(state_n)

    # Update the scheduler
    scheduler.step()

    return local_metrics_avg, global_model, scheduler

def initialize_control_variates(global_model: torch.nn.Module):
    return {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
