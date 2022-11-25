import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from pathlib import Path

__all__ = ["save_seg"]

def save_seg(img, seg, save_path):
    """
    Inputs:
        img: (h, w, 3)
        seg: (h, w)
        save_path: string
    """
    assert img.shape[0] == seg.shape[0] and img.shape[1] == seg.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=[20, 20])
    axes[0].imshow(img)
    axes[1].imshow(seg, cmap="gray")
    mask = cv2.merge([seg, seg, seg])
    mask = mask / np.max(mask)
    masked_img = img.copy().astype(np.float32)
    masked_img *= mask
    masked_img = np.clip(masked_img, 0, 255).astype(np.uint8)
    axes[2].imshow(masked_img)
    # assert Path(save_path).is_file(), f"save_path should be a filepath, but got: {save_path}"
    if not Path(save_path).parent.exists():
        print(f"mkdir: {Path(save_path).parent}")
        Path(save_path).parent.mkdir(parents=True)

    plt.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)

    