from datasets import get_single_frame, plot_frame_comparison
from example_methods import NonLocalMeans, MedianFilter
from pathlib import Path
import pyFAI

_, frame = get_single_frame(
    Path("/scratch/nal89286/data/Denoising/i22-629820.nxs"),
    "/entry1/Pilatus2M_WAXS/data",
    1,
    "entry1/instrument/detector/count_time",
)

model = MedianFilter()

result = model.forward(frame)

plot_frame_comparison(frame.squeeze(), result.squeeze())
