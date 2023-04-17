import numpy as np
import ray


def fed_avg_baseline(aggregator, client_trainers, client_dataset_refs, client_num_per_round, global_model, round_idx,
                     scheduler, device):
    remote_steps = []
    # Select random clients for each round
    sampled_clients_idx = np.random.choice(len(client_dataset_refs), client_num_per_round, replace=False)
    for i, client_trainer in enumerate(client_trainers):
        # Update the remote client_trainer with the latest global model and scheduler state
        client_trainer.update.remote(global_model.state_dict(), scheduler.state_dict())

        # Perform a remote training step on the client_trainer
        remote_step = client_trainer.step.remote(sampled_clients_idx[i],
                                                 client_dataset_refs[sampled_clients_idx[i]],
                                                 round_idx,
                                                 device=device)
        remote_steps.append(remote_step)
    # Retrieve remote steps results
    updates, weights, local_metrics = zip(*ray.get(remote_steps))
    # Calculate the average local metrics
    local_metrics_avg = {key: sum(d[key] for d in local_metrics if d[key]) / len(local_metrics) for key in
                         local_metrics[0]}
    # Update the global model using the aggregator
    state_n = aggregator.step(updates, weights, round_idx)
    global_model.load_state_dict(state_n)
    scheduler.step()
    return local_metrics_avg, global_model, scheduler
