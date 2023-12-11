import numpy
import cv2
from torch import Tensor, from_numpy


def to_numpy_unit8(input: Tensor) -> numpy.uint8:
    return numpy.uint8(input.squeeze().numpy())


class NonLocalMeans:
    def forward(self, x: Tensor) -> Tensor:
        result = cv2.fastNlMeansDenoising(to_numpy_unit8(x), None)
        return from_numpy(result)


class MedianFilter:
    def __init__(self) -> None:
        self.ksize = 5

    def forward(self, x: Tensor) -> Tensor:
        result = cv2.medianBlur(to_numpy_unit8(x), ksize=self.ksize)
        return from_numpy(result)
