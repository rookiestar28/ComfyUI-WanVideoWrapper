"""SCAIL-Pose2 replacement noise-mask conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCAIL_POSE2_CONDITION_MODE_ATTR = "scail_pose2_condition_mode"
SCAIL_POSE2_MASK_ROLE_ATTR = "scail_pose2_mask_role"
SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE = "replacement_denoise_mask"


@dataclass(frozen=True)
class NoiseMaskLatentContract:
    original_shape: tuple[int, ...]
    prepared_shape: tuple[int, ...]
    latent_shape: tuple[int, int, int]
    channel_count: int
    interpolation_mode: str
    scail_pose2_replacement: bool
    frame_policy: str
    subject_ratio: float
    preserve_ratio: float

    def to_log_string(self) -> str:
        return (
            "noise_mask_latent_contract "
            f"original_shape={self.original_shape} "
            f"prepared_shape={self.prepared_shape} "
            f"latent_shape={self.latent_shape} "
            f"channels={self.channel_count} "
            f"mode={self.interpolation_mode} "
            f"scail_pose2_replacement={self.scail_pose2_replacement} "
            f"frame_policy={self.frame_policy} "
            f"subject_ratio={self.subject_ratio:.6f} "
            f"preserve_ratio={self.preserve_ratio:.6f}"
        )


def is_scail_pose2_replacement_noise_mask(noise_mask: Any) -> bool:
    return (
        getattr(noise_mask, SCAIL_POSE2_CONDITION_MODE_ATTR, None) == "replacement"
        and getattr(noise_mask, SCAIL_POSE2_MASK_ROLE_ATTR, None)
        == SCAIL_POSE2_REPLACEMENT_DENOISE_MASK_ROLE
    )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in getattr(value, "shape", ()))


def _positive_latent_shape(latent_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    parsed = tuple(int(part) for part in latent_shape)
    if len(parsed) != 3 or any(part <= 0 for part in parsed):
        raise ValueError("latent_shape must be a positive (frames, height, width) tuple")
    return parsed


def _positive_channel_count(channel_count: Any) -> int:
    parsed = int(channel_count)
    if parsed <= 0:
        raise ValueError("channel_count must be positive")
    return parsed


def resize_noise_mask_for_latents(
    noise_mask: Any,
    *,
    latent_shape: tuple[int, int, int],
    channel_count: Any,
    start_latent: int | None = None,
    end_latent: int | None = None,
    source_latent_frame_count: int | None = None,
) -> tuple[Any, NoiseMaskLatentContract]:
    """Resize a sampler `noise_mask` to `[1, C, T, H, W]` latent mask shape."""

    import torch.nn.functional as F

    target_frames, target_height, target_width = _positive_latent_shape(latent_shape)
    channels = _positive_channel_count(channel_count)
    original_shape = _shape_tuple(noise_mask)
    scail_pose2_replacement = is_scail_pose2_replacement_noise_mask(noise_mask)
    prepared = noise_mask

    if len(prepared.shape) == 4:
        prepared = prepared.squeeze(1)
    if len(prepared.shape) != 3:
        raise ValueError("noise_mask must have shape [T,H,W] or [T,1,H,W]")

    frame_policy = "direct"
    if (
        start_latent is not None
        and end_latent is not None
        and source_latent_frame_count is not None
        and int(prepared.shape[0]) != int(source_latent_frame_count)
    ):
        source_frames = int(source_latent_frame_count)
        if source_frames <= 0:
            raise ValueError("source_latent_frame_count must be positive")
        prepared = _resize_prepared_mask(
            prepared,
            target_frames=source_frames,
            target_height=target_height,
            target_width=target_width,
            scail_pose2_replacement=scail_pose2_replacement,
        )
        prepared = prepared[int(start_latent):int(end_latent)]
        frame_policy = (
            f"resize_full_{source_frames}_then_slice_"
            f"{int(start_latent)}_{int(end_latent)}"
        )
    elif prepared.shape[0] < target_frames:
        repeat_count = max(1, target_frames // int(prepared.shape[0]))
        if repeat_count > 1:
            prepared = prepared.repeat(repeat_count, 1, 1)
            frame_policy = f"repeat_x{repeat_count}"
        else:
            frame_policy = "interpolate_time"
    elif start_latent is not None and end_latent is not None:
        prepared = prepared[int(start_latent):int(end_latent)]
        frame_policy = f"slice_{int(start_latent)}_{int(end_latent)}"

    prepared_shape = _shape_tuple(prepared)
    interpolation_mode = "nearest" if scail_pose2_replacement else "trilinear"
    resized_3d = _resize_prepared_mask(
        prepared,
        target_frames=target_frames,
        target_height=target_height,
        target_width=target_width,
        scail_pose2_replacement=scail_pose2_replacement,
    )
    resized = resized_3d.unsqueeze(0).unsqueeze(0)
    subject_ratio = float(resized.float().mean().item())
    contract = NoiseMaskLatentContract(
        original_shape=original_shape,
        prepared_shape=prepared_shape,
        latent_shape=(target_frames, target_height, target_width),
        channel_count=channels,
        interpolation_mode=interpolation_mode,
        scail_pose2_replacement=scail_pose2_replacement,
        frame_policy=frame_policy,
        subject_ratio=subject_ratio,
        preserve_ratio=1.0 - subject_ratio,
    )
    return resized.repeat(1, channels, 1, 1, 1), contract


def _resize_prepared_mask(
    prepared: Any,
    *,
    target_frames: int,
    target_height: int,
    target_width: int,
    scail_pose2_replacement: bool,
) -> Any:
    import torch.nn.functional as F

    view = prepared.unsqueeze(0).unsqueeze(0)
    if scail_pose2_replacement:
        resized = F.interpolate(
            view,
            size=(target_frames, target_height, target_width),
            mode="nearest",
        )
        resized = (resized >= 0.5).to(dtype=prepared.dtype)
    else:
        resized = F.interpolate(
            view,
            size=(target_frames, target_height, target_width),
            mode="trilinear",
            align_corners=False,
        )
    return resized.squeeze(0).squeeze(0)
