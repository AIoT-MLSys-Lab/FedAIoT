import torch


class ParameterDict:
    def __init__(self, parameter_dict):
        for key, value in parameter_dict.items():
            setattr(self, key, value)


def pack_pathway_output(frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        1,
        torch.linspace(
            0, frames.shape[1] - 1, frames.shape[1] // 4
        ).long(),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list
