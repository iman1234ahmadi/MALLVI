from typing import List
import numpy as np
from dataclasses import dataclass
from enum import Enum
from state import GraphState, GraspPoint3D

class ProjectorMode(Enum):
    """Enumeration of projector operation modes"""
    REAL_WORLD = "real_world"
    VIMA = "vima"

@dataclass
class ProjectorConfig:
    """Configuration for projector node operation mode"""
    mode: ProjectorMode = ProjectorMode.VIMA

    # pix_size = 0.003125
    pix_size = 0.00078125
    bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])

    # VIMA mode parameters (linear transformations for x and y)
    vima_x_scale: float = pix_size
    vima_x_offset: float = bounds[0, 0]
    vima_y_scale: float = pix_size
    vima_y_offset: float = bounds[1, 0]

def projector_node(state: GraphState, config: ProjectorConfig = None) -> dict:
    """
    Project 2D grasp points to 3D world coordinates for bird's-eye view.

    Inputs:
        grasp_points: List[GraspPoint] from segmentor (u, v, label)
        image: PIL.Image for dimensions
        camera_matrix: 3x3 camera matrix (not used in bird's-eye view)
        rotation_matrix: 3x3 rotation matrix (not used in bird's-eye view)
        translation_vector: 3x1 translation vector (not used in bird's-eye view)

    Output:
        List of GraspPoint3D with world coordinates (x, y, z)
    """
    # Use default config if none provided (default to VIMA mode since depth is not supported)
    if config is None:
        config = ProjectorConfig(mode=ProjectorMode.VIMA)

    # Get inputs from state
    grasp_points = state.get('grasp_points', [])
    depth_image = state.get('depth_image')

    grasp_points_3d: List[GraspPoint3D] = []

    if config.mode == ProjectorMode.REAL_WORLD:
        # Check if depth information is available
        if depth_image is None:
            print("⚠️  Depth image not available, falling back to VIMA mode")
            config.mode = ProjectorMode.VIMA

        if config.mode == ProjectorMode.REAL_WORLD:
            # Get camera parameters from state
            K = state.get('camera_matrix')
            R = state.get('rotation_matrix')
            t = state.get('translation_vector')

            if K is None or R is None or t is None:
                print("⚠️  Camera parameters not available, falling back to VIMA mode")
                config.mode = ProjectorMode.VIMA

    if config.mode == ProjectorMode.REAL_WORLD:
        # Calculate inverse of rotation matrix for real-world mode
        R_inv = np.linalg.inv(R)

        for point in grasp_points:
            u, v, label = point['u'], point['v'], point['label']

            # Get depth at this point (ensure coordinates are integers and within bounds)
            if depth_image is not None and 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                depth = depth_image[v, u]
            else:
                # If out of bounds or no depth, use average depth or skip
                depth = np.mean(depth_image) if depth_image is not None else 1.0

            # Back-project from 2D to 3D (camera coordinates)
            # Convert to homogeneous coordinates
            uv_hom = np.array([u, v, 1.0])

            # Calculate normalized camera coordinates
            K_inv = np.linalg.inv(K)
            point_cam = depth * (K_inv @ uv_hom)

            # Transform to world coordinates
            point_world = R_inv @ (point_cam - t)

            # Create 3D grasp point
            grasp_points_3d.append({
                "x": float(point_world[0]),
                "y": float(point_world[1]),
                "z": float(point_world[2]),
                "label": label
            })

    elif config.mode == ProjectorMode.VIMA:
        # VIMA mode: simple linear transformation of x and y, z always 0
        for point in grasp_points:
            u, v, label = point['u'], point['v'], point['label']

            # Apply linear transformations to u and v coordinates
            x = config.vima_x_scale * v + config.vima_x_offset
            y = config.vima_y_scale * u + config.vima_y_offset
            z = 0.0  # Always 0 in VIMA mode

            # Create 3D grasp point
            grasp_points_3d.append({
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "label": label
            })

    # For demo purposes, print the 3D points
    print(f"Projector ({config.mode.value}) generated {len(grasp_points_3d)} 3D grasp points:")
    for i, point in enumerate(grasp_points_3d):
        print(f"  {i+1}. {point['label']}: ({point['x']:.2f}, {point['y']:.2f}, {point['z']:.2f})")

    return {
        'grasp_points_3d': grasp_points_3d
    }
