from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import hdf5plugin
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm  # noqa: F401
import numpy
import torch
from torch import Tensor, dtype, from_numpy


def get_single_frame(
    hdf_path: Path,
    data_key: str,
    idx: int,
    count_time_key: Optional[str],
    data_type: dtype = torch.float32,
) -> Tuple[Any, Tensor]:
    hdf_file = h5py.File(hdf_path)
    hdf_dataset = hdf_file[data_key]
    count_time = None
    if count_time_key:
        count_time = hdf_file[count_time_key][0]
    return (
        count_time,
        from_numpy(numpy.array(hdf_dataset[:, idx, :, :])).type(data_type),
    )


def plot_frame_comparison(original_frame: Tensor, denoised_frame: Tensor) -> None:
    vmin = 0.01
    vmax = 1000000
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(original_frame, cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax[0].set_title("original")
    ax[1].imshow(denoised_frame, cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax[1].set_title("denoised")
    ax[2].imshow((original_frame - denoised_frame), norm=LogNorm(vmin=vmin, vmax=vmax))
    ax[2].set_title("dif")
    plt.tight_layout()
    plt.show()
