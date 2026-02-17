"""
Configuration Classes for Hydra
===============================

Data classes for Hydra configuration management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Literal


@dataclass
class VIMAConfig:
    """Configuration for VIMA environment"""
    task_name: Optional[str] = "instruction_following/visual_manipulation"
    modalities: List[str] = field(default_factory=lambda: ["rgb"])
    debug: bool = False
    display_debug_window: bool = True
    show_gui: bool = True
    hide_arm_rgb: bool = False  # Hide robot arm for clean observations
    gui_delay: float = 0.1
    action_delay: float = 0.5
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: int = 60
    simulation_frequency: int = 240
    control_frequency: int = 30
    max_action_attempts: int = 3
    action_timeout: float = 10.0
    log_level: str = "INFO"
    enable_action_logging: bool = True
    enable_observation_logging: bool = False
    save_observation_images: bool = True  # Save images when getting observations
    observation_image_dir: str = "observations"  # Directory to save observation images

    # Default positions for basic tasks
    default_targets: Dict[str, Any] = field(default_factory=lambda: {
        "pick_position": [0.5, 0.0, 0.1],
        "place_position": [0.5, 0.2, 0.1],
        "orientation": [0.0, 0.0, 0.0]
    })


@dataclass
class GrounderConfig:
    """Configuration for the grounder node"""
    # Core configuration
    grounding_mode: Literal["owl", "dino", "both", "simple"] = "both"

    # OWLv2 Configuration
    owl_model_id: str = "google/owlv2-base-patch16-ensemble"
    owl_model_path: Optional[str] = "models/owl/"
    owl_threshold: float = 0.1

    # GroundingDINO Configuration
    dino_model_id: str = "IDEA-Research/grounding-dino-base"
    dino_model_path: Optional[str] = "models/dino/"
    dino_threshold: float = 0.40
    dino_text_threshold: float = 0.3
    dino_pseudo_score: float = 0.60

    # Detection Processing
    iou_merge_threshold: float = 0.30
    fallback_fraction: float = 0.30

    # Robustness Configuration
    auto_fallback_to_simple: bool = True

    # Device Configuration
    device: Optional[str] = "cuda"


@dataclass
class SegmentorConfig:
    """Configuration for the segmentor node"""
    # Core configuration
    backend: Literal["sam", "box_only", "custom"] = "box_only"
    points_per_box: int = 1
    min_area: int = 10
    dt_suppress_radius: int = 8
    axis_order: Literal["xy", "uv"] = "xy"
    point_mode: Literal[
        "auto", "dt", "centroid", "bbox_center", "topmost",
        "bottommost", "leftmost", "rightmost", "grid"
    ] = "auto"

    # SAM-specific configuration (only for "sam" backend)
    sam_model_type: Literal["sam", "sam2"] = "sam"  # Choose between SamModel and Sam2Model
    sam_model_name: str = "facebook/sam-vit-base"
    sam_model_path: Optional[str] = "models/sam/"
    sam2_model_name: str = "facebook/sam2.1-hiera-large"
    sam2_model_path: Optional[str] = "models/sam2/"

    # Custom backend configuration (only for "custom" backend)
    custom_backend: Optional[Any] = None
    custom_backend_fn: Optional[Any] = None

    # Point mode overrides for specific labels
    point_mode_map: Dict[str, str] = field(default_factory=dict)

    # Device Configuration
    device: Optional[str] = "cuda"


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing"""
    enable_preprocessing: bool = True
    normalization: bool = True
    resize_images: bool = False
    target_size: List[int] = field(default_factory=lambda: [640, 480])


@dataclass
class ActionPlanningConfig:
    """Configuration for action planning"""
    max_actions_per_task: int = 10
    enable_orientation_control: bool = True
    rotation_degrees_limit: int = 180
    position_tolerance: float = 0.01


@dataclass
class TaskConfig:
    """Configuration for task management"""
    default_prompt: str = "Pick up the red block and place it on the table."
    enable_multi_object: bool = False
    max_prompts_per_task: int = 5


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    enable_realtime_display: bool = True
    show_depth_overlay: bool = False
    show_grasp_points: bool = True
    update_frequency: int = 10


@dataclass
class PerformanceConfig:
    """Configuration for performance settings"""
    enable_multithreading: bool = False
    max_workers: int = 4
    memory_limit_gb: int = 8


@dataclass
class PipelineConfig:
    """Main pipeline configuration"""
    pipeline_name: str = "robotic_manipulation_pipeline"
    enable_logging: bool = True
    log_directory: str = "logs"
    node_timeout: float = 30.0
    max_retries: int = 3
    enable_node_profiling: bool = False

    # Nested configurations
    vima: VIMAConfig = field(default_factory=VIMAConfig)
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    segmentor: SegmentorConfig = field(default_factory=SegmentorConfig)
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    action_planning: ActionPlanningConfig = field(default_factory=ActionPlanningConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


# Default configurations
def get_default_vima_config() -> VIMAConfig:
    """Get default VIMA configuration for GUI display"""
    return VIMAConfig(
        task_name=None,
        modalities=["rgb"],
        debug=False,
        display_debug_window=True,
        show_gui=True,
        hide_arm_rgb=False,
        gui_delay=0.1,
        action_delay=0.5,
        enable_action_logging=True,
        enable_observation_logging=False
    )


def get_default_grounder_config() -> GrounderConfig:
    """Get default grounder configuration"""
    return GrounderConfig(
        grounding_mode="simple",
        auto_fallback_to_simple=True,
        device="cuda"
    )


def get_default_segmentor_config() -> SegmentorConfig:
    """Get default segmentor configuration"""
    return SegmentorConfig(
        backend="box_only",
        sam_model_type="sam",
        device="cuda"
    )


def get_default_pipeline_config() -> PipelineConfig:
    """Get default pipeline configuration"""
    return PipelineConfig(
        enable_logging=True,
        vima=get_default_vima_config(),
        grounder=get_default_grounder_config(),
        segmentor=get_default_segmentor_config(),
        visualization=VisualizationConfig(enable_realtime_display=True)
    )
