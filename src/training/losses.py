"""Loss computation that handles models with auxiliary outputs.

Every model returns plain logits ``[N, num_classes]``, except GoogLeNet with
``aux_logits=True`` in training mode, which returns the named tuple
``GoogLeNetOutputs(logits, aux1, aux2)``. Tuple outputs are treated as
``(main_logits, *auxiliary_logits)``.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

import torch

ModelOutput = Union[torch.Tensor, Sequence[torch.Tensor]]


def split_outputs(outputs: ModelOutput) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Split a model output into ``(main_logits, [auxiliary_logits, ...])``."""
    if isinstance(outputs, torch.Tensor):
        return outputs, []
    if isinstance(outputs, (tuple, list)) and outputs and all(isinstance(o, torch.Tensor) for o in outputs):
        return outputs[0], list(outputs[1:])
    raise TypeError(f"Unsupported model output type: {type(outputs).__name__}")


def classification_loss(
    outputs: ModelOutput,
    targets: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    aux_weight: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, main_logits)``.

    ``loss = criterion(main) + aux_weight * sum(criterion(aux_i))``. The 0.3
    default is the auxiliary weight from the GoogLeNet paper. Accuracy should
    always be computed from ``main_logits`` only.
    """
    logits, aux_logits = split_outputs(outputs)
    loss = criterion(logits, targets)
    for aux in aux_logits:
        loss = loss + aux_weight * criterion(aux, targets)
    return loss, logits
