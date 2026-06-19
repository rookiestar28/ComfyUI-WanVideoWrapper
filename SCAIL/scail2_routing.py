from __future__ import annotations

from typing import Any, Callable, MutableMapping


SCAIL_V1_EMBEDS_KEY = "scail_embeds"
SCAIL2_EMBEDS_KEY = "scail2_embeds"
SCAIL2_MODEL_ARG = "scail2_input"


def prepare_scail2_data(
    image_embeds: MutableMapping[str, Any],
    *,
    dict_to_device: Callable[[dict[str, Any], Any, Any], dict[str, Any]],
    device: Any,
    dtype: Any,
) -> dict[str, Any] | None:
    """Extract native SCAIL-2 embeds and move copied data to sampler device."""

    scail2_embeds = image_embeds.get(SCAIL2_EMBEDS_KEY)
    if scail2_embeds is None:
        return None

    if image_embeds.get(SCAIL_V1_EMBEDS_KEY) is not None:
        raise ValueError(
            "SCAIL-2 native scail2_embeds cannot be combined with v1 scail_embeds"
        )

    if not isinstance(scail2_embeds, dict):
        raise TypeError("image_embeds['scail2_embeds'] must be a dict")

    return dict_to_device(scail2_embeds.copy(), device, dtype)


def scail2_context_window_input(
    scail2_data: dict[str, Any] | None,
    context_window: Any,
) -> dict[str, Any] | None:
    if scail2_data is None:
        return None

    # SCAIL-2 mask/reference token alignment must be sliced deliberately; the
    # v1 pose-only shortcut is not safe for the native v2 payload.
    if context_window is not None:
        raise ValueError(
            "SCAIL-2 native scail2_embeds do not support wrapper context windows yet"
        )

    return scail2_data


def add_scail2_model_param(
    base_params: dict[str, Any],
    scail2_data_in: dict[str, Any] | None,
) -> dict[str, Any]:
    if scail2_data_in is not None:
        base_params[SCAIL2_MODEL_ARG] = scail2_data_in
    return base_params
