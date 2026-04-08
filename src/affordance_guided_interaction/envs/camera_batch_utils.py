from __future__ import annotations

from torch import Tensor


def align_camera_tensor_batch(
    tensor: Tensor,
    *,
    batch_size: int,
    name: str,
) -> Tensor:
    """Align a batched camera tensor to the expected environment batch size."""
    if tensor.shape[0] < batch_size:
        raise RuntimeError(
            f"{name} batch smaller than expected visual batch: "
            f"shape={tuple(tensor.shape)} batch_size={batch_size}"
        )
    if tensor.shape[0] != batch_size:
        tensor = tensor[:batch_size]
    return tensor


def align_camera_pose_batch(
    pos_w: Tensor,
    quat_w_ros: Tensor,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Align camera pose tensors to the visual batch size."""
    pos_w = align_camera_tensor_batch(pos_w, batch_size=batch_size, name="camera pos")
    quat_w_ros = align_camera_tensor_batch(
        quat_w_ros,
        batch_size=batch_size,
        name="camera quat",
    )
    return pos_w, quat_w_ros
