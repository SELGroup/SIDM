import numpy as np
import visdom
import torch

# Initialize visdom client
vis = visdom.Visdom()

# Create a random image and show it
image = np.random.rand(3, 256, 256)
vis.image(image, win='image1')

# # Create a random video and show it
# video = torch.randn(3, 4, 3, 64, 64)  # (C, T, H, W)
# vis.video(video, win='video1')
# Create a random video and show it
video = torch.randn(1, 30, 64, 64)  # (C, T, H, W)
vis.video(video, win='video2')