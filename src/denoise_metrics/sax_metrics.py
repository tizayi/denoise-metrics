from typing import Protocol
from torch import Tensor
import pyFAI


class DenoiseMetric(Protocol):
    def measure(input: Tensor):
        ...


ai = pyFAI.AzimuthalIntegrator()
