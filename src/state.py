from typing import TypedDict, List, Dict, Optional, Tuple, Any

from numpy import float64, ndarray
from PIL import Image

class Detection(TypedDict):
    bounding_box: tuple[int, int, int, int]
    label: str
    score: float64

# Define grasp point type
class GraspPoint(TypedDict):
    u: int  # x-coordinate
    v: int  # y-coordinate
    label: str  # From grounder detection

# Define 3D point type
class GraspPoint3D(TypedDict):
    x: float
    y: float
    z: float
    label: str

class PickPlace(TypedDict, total=False):
    pick: GraspPoint3D
    place: GraspPoint3D

class ThinkerOutput(TypedDict):
    decision: str
    # List of (pick_position, place_position) tuples in world coordinates
    chosen_grasp_points: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]]
    reasoning: str
    # Rotation extracted from prompt (in degrees)
    rotation_degrees: Optional[List[float]]

class Action(TypedDict):
    # Positions are in world coordinates
    pick_position: Tuple[float, float, float]
    place_position: Tuple[float, float, float]
    # Orientations are Euler angles in radians: (roll, pitch, yaw)
    pick_orientation: Tuple[float, float, float]
    place_orientation: Tuple[float, float, float]

class ActorOutput(TypedDict):
    task_done: bool
    execution_log: str
    actions: List[Action]

class ReflectionResult(TypedDict):
    task_complete: bool
    verification_result: str
    confidence: float

class GraphState(TypedDict):
    """State definition with strict typing"""
    original_prompt: str
    initial_decomposition_done: bool  # New flag to track decomposition status
    decomposed_prompts: List[str]          # From decomposer
    queue: List[str]                       # Prompt processing queue
    current_prompt: Optional[str]          # Currently processed prompt
    object_of_interest: Optional[str]          # Per-prompt output
    not_object_of_interest: Optional[str]     # Per-prompt output
    results: Dict[str, dict]               # Prompt → {object: str, not_objects: List[str]}
    image: Image.Image  # PIL image
    grounder_output: List[Detection]  # From grounder node
    depth_image: Optional[ndarray]  # Depth map (H, W) array - VIMA doesn't support depth
    camera_matrix: ndarray  # 3x3 camera matrix
    rotation_matrix: ndarray  # 3x3 rotation matrix
    translation_vector: ndarray  # 3x1 translation vector
    grasp_points: List[GraspPoint]  # From segmentor node
    grasp_points_3d: List[GraspPoint3D]  # From projector node
    thinker_output: Dict[str, ThinkerOutput]  # Prompt → Thinker output
    actor_output: Dict[str, ActorOutput] # Prompt -> Actor output
    reflection_output: Dict[str, ReflectionResult]  # Prompt → Reflection result
    should_terminate: bool  # Global termination flag
    multi_object: bool # Whether the current prompt is for multiple objects
    # VIMA-specific fields
    vima_prompt_assets: Optional[Dict[str, Any]]  # Assets from VIMA task
    vima_metadata: Optional[Dict[str, Any]]  # Metadata from VIMA environment
    vima_interface: Optional[Any]  # VIMA interface instance for actor access
    oracle_action: Optional[Any]  # Oracle action for debugging
