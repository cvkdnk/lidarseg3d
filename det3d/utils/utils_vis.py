import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

def visualize_point_cloud_bev(
    point_cloud: np.ndarray,
    features: np.ndarray,
    save_path: str,
    bev_range: tuple = (-40, -40, 40, 40),
    bev_resolution: tuple = (256, 256),
    fixed_min: float = None,
    fixed_max: float = None,
    color_map: str = 'jet',
    annotation: str = None
) -> None:
    """
    Visualize point cloud features in BEV (Bird's Eye View) image.

    Args:
        point_cloud (np.ndarray): The point cloud coordinates, expected shape (N, D).
        features (np.ndarray): Point cloud features, shape (N, C) where C can be 1 or 3.
        save_path (str): Path to save the resulting BEV image.
        bev_range (tuple): Visualization range in BEV, format (min_x, min_y, max_x, max_y).
        bev_resolution (tuple): Resolution of the BEV image (width, height).
        fixed_min (float): Fixed minimum value for feature scaling, defaults to None.
        fixed_max (float): Fixed maximum value for feature scaling, defaults to None.
        color_map (str): Color map type for feature visualization.
        annotation (str): Optional text annotation to overlay on the image.

    Returns:
        None
    """
    # Extract xyz coordinates and check dimensions
    xyz = point_cloud[:, :3]
    assert xyz.shape[0] == features.shape[0], "Mismatch in point cloud and features dimensions."
    features = features.reshape(features.shape[0], -1)

    # Calculate the pixel position in the BEV image
    px = ((xyz[:, 0] - bev_range[0]) / (bev_range[2] - bev_range[0]) * bev_resolution[0]).astype(int)
    py = ((xyz[:, 1] - bev_range[1]) / (bev_range[3] - bev_range[1]) * bev_resolution[1]).astype(int)
    px = np.clip(px, 0, bev_resolution[0] - 1)
    py = np.clip(py, 0, bev_resolution[1] - 1)

    if features.shape[1] == 3:
        bev_image = np.zeros((bev_resolution[1], bev_resolution[0], 3), dtype=np.uint8)
    else:
        bev_image = np.zeros((bev_resolution[1], bev_resolution[0]), dtype=np.float32)


    if features.shape[1] == 1:
        bev_image[py, px] = features[:, 0]

        if fixed_min is not None and fixed_max is not None:
            bev_image = Normalize(vmin=fixed_min, vmax=fixed_max)(bev_image)
        else:
            bev_image = Normalize(vmin=bev_image.min(), vmax=bev_image.max())(bev_image)

        bev_image = plt.get_cmap(color_map)(bev_image)[:, :, :3] 
        bev_image = (bev_image * 255).astype(np.uint8) 

    else:
        bev_image[py, px, :] = features[:, :3] 
        bev_image = (bev_image * 255).astype(np.uint8)

    if annotation is not None:
        cv2.putText(bev_image, annotation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imsave(save_path, bev_image)
    print(f"Saved BEV image at {save_path}")