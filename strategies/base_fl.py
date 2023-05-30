import numpy as np
import ray
from tqdm import tqdm


def distributed_fedavg(aggregator,
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
    # Initialize lists to store updates, weights, and local metrics
    all_updates, all_weights, all_local_metrics = [], [], []

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
                                                                       device=device)
            else:
                remote_step = client_trainer.step.remote(sampled_clients_idx[idx],
                                                         client_dataset_refs[sampled_clients_idx[idx]],
                                                         round_idx,
                                                         device=device)
            remote_steps.append(remote_step)

        # Retrieve remote steps results
        print(f"length of steps: {len(remote_steps)}")
        updates, num_client_samples, local_metrics = zip(*ray.get(remote_steps))

        # Add the results to the overall lists
        for u, n, l in zip(updates, num_client_samples, local_metrics):
            if n > 0:
                all_updates.append(u)
                all_weights.append(n)
                all_local_metrics.append(l)
        # torch.cuda.empty_cache()

    # Calculate the average local metrics
    local_metrics_avg = {key: sum(metric[key] for metric in all_local_metrics if metric[key]) / len(all_local_metrics)
                         for key in all_local_metrics[0]}

    print(all_local_metrics)

    # Update the global model using the aggregator
    state_n = aggregator.step(all_updates, all_weights, round_idx)
    global_model.load_state_dict(state_n)

    # Update the scheduler
    scheduler.step()

    return local_metrics_avg, global_model, scheduler


def basic_fedavg(aggregator,
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
    # Initialize lists to store updates, weights, and local metrics
    all_updates, all_weights, all_local_metrics = [], [], []

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
            client_trainer.update(global_model.state_dict(), scheduler.state_dict())

            # Perform a remote training step on the client_trainer
            if precision != 'float32':
                remote_step = client_trainer.step_low_precision(sampled_clients_idx[idx],
                                                                client_dataset_refs[sampled_clients_idx[idx]],
                                                                round_idx,
                                                                precision,
                                                                device=device)
            else:
                remote_step = client_trainer.step(sampled_clients_idx[idx],
                                                  client_dataset_refs[sampled_clients_idx[idx]],
                                                  round_idx,
                                                  device=device)
            remote_steps.append(remote_step)

        # Retrieve remote steps results
        print(f"length of steps: {len(remote_steps)}")
        updates, num_client_samples, local_metrics = zip(*remote_steps)

        # Add the results to the overall lists
        all_updates.extend(updates)
        all_weights.extend(num_client_samples)
        all_local_metrics.extend(local_metrics)
        # torch.cuda.empty_cache()

    # Calculate the average local metrics
    local_metrics_avg = {key: sum(metric[key] for metric in all_local_metrics if metric[key]) / len(all_local_metrics)
                         for key in all_local_metrics[0]}
    for m in all_local_metrics:
        if m['Local Loss'] == 0 or np.isnan(m['Local Loss']):
            break
    print(all_local_metrics)

    # Update the global model using the aggregator
    state_n = aggregator.step(all_updates, all_weights, round_idx)
    global_model.load_state_dict(state_n)

    # Update the scheduler
    scheduler.step()

    return local_metrics_avg, global_model, scheduler
