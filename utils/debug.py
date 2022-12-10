import numpy as np
import matplotlib.pyplot as plt


__all__ = ['debug_model_input']

def debug_model_input(self, img, gt_seg):
    if self.rank == 0:
        rnd_i = np.random.randint(low=0, high=img.size(0))
        img = img.permute(0, 2, 3, 1)
        gt_seg = gt_seg.permute(0, 2, 3, 1).squeeze()
        img = img[rnd_i].detach().cpu().numpy()
        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        seg = gt_seg[rnd_i].detach().cpu().numpy().astype(np.uint8)
        fig, axes = plt.subplots(2, 1, figsize=[12, 12])
        axes[0].imshow(img)
        axes[1].imshow(seg, cmap='gray')
        plt.savefig(f"./debug_{rnd_i}_rank_{self.rank}.png")