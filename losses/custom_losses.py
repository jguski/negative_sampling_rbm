from pykeen.losses import DeltaPointwiseLoss
import torch
from torch.nn import LogSigmoid
from typing import Any, ClassVar, Mapping

class ShiftLogLoss(DeltaPointwiseLoss):

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        shift = dict(type=int, low=0, high=25, step=1)
    )

    def __init__(self, reduction: str = "sum", shift = 11) -> None:
        super().__init__(margin=0.0, reduction=reduction, margin_activation=LogSigmoid())
        self.shift = shift

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:

        return super().forward(logits=(self.shift+logits), labels=labels)

    