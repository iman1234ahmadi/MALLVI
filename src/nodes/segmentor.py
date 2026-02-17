from __future__ import annotations
from state import GraphState, Detection, GraspPoint
from typing import Dict, List, Tuple, Optional, Protocol, runtime_checkable
import numpy as np
from PIL import Image as PIL_Image
from config.config_classes import SegmentorConfig

# Check for optional dependencies
try:
    import torch
    from transformers import SamModel, SamProcessor, Sam2Model, Sam2Processor
    _HAS_SAM = True
    _HAS_SAM2 = True
except ImportError:
    try:
        # Try importing just SAM if SAM2 is not available
        from transformers import SamModel, SamProcessor
        _HAS_SAM = True
        _HAS_SAM2 = False
        Sam2Model = None
        Sam2Processor = None
    except ImportError:
        _HAS_SAM = False
        _HAS_SAM2 = False
        SamModel = None
        SamProcessor = None
        Sam2Model = None
        Sam2Processor = None
        torch = None

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None

@runtime_checkable
class MaskBackend(Protocol):
    """Protocol defining the interface for mask generation backends"""
    def segment(
        self,
        image: PIL_Image,
        boxes: List[Tuple[int, int, int, int]],
        labels: List[str],
        scores: Optional[List[float]] = None
    ) -> List[Optional[np.ndarray]]:
        """
        Generate masks for given bounding boxes.

        Args:
            image: Input image
            boxes: List of (x1, y1, x2, y2) bounding boxes
            labels: List of object labels
            scores: Optional list of confidence scores

        Returns:
            List of masks (same length as boxes), None for failed masks
        """
        ...

# Helper functions
def _clip_box(b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Ensure bounding box coordinates are properly ordered"""
    x1, y1, x2, y2 = map(int, b)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2

def _center(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Calculate center point of bounding box"""
    x1, y1, x2, y2 = b
    return (x1 + x2) // 2, (y1 + y2) // 2

def _grid_points(b: Tuple[int, int, int, int], k: int) -> List[Tuple[int, int]]:
    """Generate grid-based points within bounding box"""
    cx, cy = _center(b)
    if k <= 1:
        return [(cx, cy)]

    x1, y1, x2, y2 = b
    s = max(2, int(np.ceil(np.sqrt(k))))
    xs = np.linspace(x1, x2, num=s, dtype=int)
    ys = np.linspace(y1, y2, num=s, dtype=int)
    pts = [(int(x), int(y)) for y in ys for x in xs]

    # Sort by distance from center, put center first
    pts.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
    if pts and pts[0] != (cx, cy):
        pts.insert(0, (cx, cy))
        # Remove duplicates while preserving order
        seen, out = set(), []
        for p in pts:
            if p not in seen:
                out.append(p)
                seen.add(p)
        pts = out

    return pts[:k]

def _apply_axis_order(pt: Tuple[int, int], order: str) -> Tuple[int, int]:
    """Apply coordinate system ordering"""
    x, y = pt
    return (x, y) if order == "xy" else (y, x)

def _to_mask2d(mask: np.ndarray) -> np.ndarray:
    """
    Normalize any mask to 2D binary (H,W) in {0,1}.

    Handles various mask shapes like (1,H,W), (H,W,1), (H,W,3/4), etc.
    """
    a = np.asarray(mask)
    a = np.squeeze(a)

    if a.ndim == 3:
        if a.shape[0] in (1, 3, 4):
            a = (a > 0).any(axis=0)
        elif a.shape[-1] in (1, 3, 4):
            a = (a > 0).any(axis=-1)
        else:
            a = a[0]

    if a.ndim != 2:
        h, w = a.shape[-2], a.shape[-1]
        a = a.reshape(h, w)

    return (a > 0).astype(np.uint8)

# Mask-based point extractors
def _mask_to_points_dt(mask: np.ndarray, k: int, suppress_r: int) -> List[Tuple[int, int]]:
    """Extract points using distance transform (finds points with maximum distance from background)"""
    m = _to_mask2d(mask)
    if m.sum() < 5:
        return []

    if _HAS_CV2:
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        pts: List[Tuple[int, int]] = []
        w = dt.copy()

        for _ in range(max(1, k)):
            y, x = np.unravel_index(np.argmax(w), w.shape)
            if w[y, x] <= 0:
                break
            pts.append((int(x), int(y)))
            cv2.circle(w, (int(x), int(y)), int(suppress_r), 0.0, -1)

        return pts

    # Fallback: return centroid if OpenCV not available
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return []
    return [(int(xs.mean()), int(ys.mean()))]

def _centroid_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Calculate centroid (center of mass) of mask"""
    m = _to_mask2d(mask)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())

def _extreme_from_mask(mask: np.ndarray, mode: str) -> Optional[Tuple[int, int]]:
    """Find extreme points (topmost, bottommost, leftmost, rightmost) of mask"""
    m = _to_mask2d(mask)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None

    if mode == "topmost":
        i = int(np.argmin(ys))
    elif mode == "bottommost":
        i = int(np.argmax(ys))
    elif mode == "leftmost":
        i = int(np.argmin(xs))
    else:  # rightmost
        i = int(np.argmax(xs))

    return int(xs[i]), int(ys[i])

def _infer_point_mode(label: str, box: Tuple[int, int, int, int],
                      mask: Optional[np.ndarray], k: int) -> str:
    """
    Automatically infer the best point selection mode based on object characteristics.

    This heuristic analyzes the object label and shape to choose the most appropriate
    grasp point selection strategy.
    """
    s = (label or "").lower()

    # Object type-based heuristics
    top_rim = ("cup", "mug", "glass", "bottle", "vase", "can", "goblet")
    cuboid = ("cube", "block", "box", "brick", "dice", "die")
    roundish = ("ball", "sphere", "apple", "orange", "tomato")

    if any(w in s for w in top_rim):
        return "topmost"  # Grasp from top rim
    if any(w in s for w in cuboid):
        return "dt" if mask is not None else ("grid" if k > 1 else "bbox_center")
    if any(w in s for w in roundish):
        return "centroid" if mask is not None else ("grid" if k > 1 else "bbox_center")

    # Handle-specific logic
    if "handle" in s:
        if mask is not None:
            m = _to_mask2d(mask)
            ys, xs = np.where(m > 0)
            if xs.size:
                cx_box = (box[0] + box[2]) / 2.0
                left_x, right_x = xs.min(), xs.max()
                return "rightmost" if (right_x - cx_box) >= (cx_box - left_x) else "leftmost"
        return "rightmost"

    # Aspect ratio-based heuristics
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = h / w

    if ar > 1.4:  # Tall objects
        return "topmost" if mask is not None else "bbox_center"
    if (1.0 / ar) > 1.6:  # Wide objects
        return "centroid" if mask is not None else ("grid" if k > 1 else "bbox_center")

    # Default fallbacks
    if mask is not None:
        return "dt"
    return "grid" if k > 1 else "bbox_center"

# Backend implementations
class SamMaskBackend:
    """SAM (Segment Anything Model) backend for high-quality mask generation"""

    def __init__(self, config: SegmentorConfig):
        self.config = config
        self.model_type = config.sam_model_type

        # Check if required model is available
        if self.model_type == "sam2":
            self.enabled = _HAS_SAM2
        else:
            self.enabled = _HAS_SAM

        if not self.enabled:
            self.model = None
            self.proc = None
            self.device = "cpu"
            return

        # Initialize model based on type
        dev = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load from custom path
        if self.model_type == "sam2":
            if self.config.sam2_model_path:
                self.model = Sam2Model.from_pretrained(config.sam2_model_path).to(dev)
                self.proc = Sam2Processor.from_pretrained(config.sam2_model_path)
            else:
                self.model = Sam2Model.from_pretrained(config.sam2_model_name).to(dev)
                self.proc = Sam2Processor.from_pretrained(config.sam2_model_name)
        else:
            if self.config.sam_model_path:
                self.model = Sam2Model.from_pretrained(config.sam_model_path).to(dev)
                self.proc = Sam2Processor.from_pretrained(config.sam_model_path)
            else:
                self.model = SamModel.from_pretrained(config.sam_model_name).to(dev)
                self.proc = SamProcessor.from_pretrained(config.sam_model_name)

        self.device = dev

    def segment(self, image: PIL_Image, boxes: List[Tuple[int, int, int, int]],
                labels: List[str], scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """Generate masks using SAM or SAM2 model"""
        if (not self.enabled) or image is None or len(boxes) == 0:
            return [None] * len(boxes)

        try:
            if self.model_type == "sam2":
                # SAM2 processing
                return self._segment_sam2(image, boxes, labels, scores)
            else:
                # Original SAM processing
                return self._segment_sam(image, boxes, labels, scores)

        except Exception:
            return [None] * len(boxes)

    def _segment_sam(self, image: PIL_Image, boxes: List[Tuple[int, int, int, int]],
                    labels: List[str], scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """Generate masks using SAM model"""
        # Convert boxes to SAM format
        boxes_float = [[[float(x1), float(y1), float(x2), float(y2)]
                       for (x1, y1, x2, y2) in boxes]]

        # Process inputs
        inputs = self.proc(images=image, input_boxes=boxes_float, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                 for k, v in inputs.items()}

        # Generate masks
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks
        post = self.proc.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu()
        )[0]

        # Convert to numpy and apply quality filters
        masks: List[Optional[np.ndarray]] = []
        for i in range(min(len(boxes), len(post))):
            m = post[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)

            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]

            # Threshold and normalize
            m01 = _to_mask2d(m > 0.5)
            m255 = (m01 * 255).astype(np.uint8)

            # Apply minimum area filter
            masks.append(m255 if int(m01.sum()) >= self.config.min_area else None)

        # Ensure we return the right number of masks
        while len(masks) < len(boxes):
            masks.append(None)

        return masks

    def _segment_sam2(self, image: PIL_Image, boxes: List[Tuple[int, int, int, int]],
                     labels: List[str], scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """Generate masks using SAM2 model"""
        # Convert boxes to SAM2 format
        boxes_float = [[[float(x1), float(y1), float(x2), float(y2)]
                       for (x1, y1, x2, y2) in boxes]]

        # Process inputs for SAM2
        inputs = self.proc(images=image, input_boxes=boxes_float, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                 for k, v in inputs.items()}

        # Generate masks
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks for SAM2
        if hasattr(self.proc, 'image_processor') and hasattr(self.proc.image_processor, 'post_process_masks'):
            # Use post-processing if available
            post = self.proc.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu()
            )[0]
        else:
            # Fallback for different SAM2 versions
            post = outputs.pred_masks.detach().cpu()

        # Convert to numpy and apply quality filters
        masks: List[Optional[np.ndarray]] = []
        for i in range(min(len(boxes), len(post))):
            m = post[i]
            m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)

            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]

            # Threshold and normalize
            m01 = _to_mask2d(m > 0.5)
            m255 = (m01 * 255).astype(np.uint8)

            # Apply minimum area filter
            masks.append(m255 if int(m01.sum()) >= self.config.min_area else None)

        # Ensure we return the right number of masks
        while len(masks) < len(boxes):
            masks.append(None)

        return masks

class BoxOnlyBackend:
    """Simple backend that only uses bounding boxes (no mask generation)"""

    def __init__(self, config: SegmentorConfig):
        self.config = config

    def segment(self, image: PIL_Image, boxes: List[Tuple[int, int, int, int]],
                labels: List[str], scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """Return None for all masks (box-only mode)"""
        return [None] * len(boxes)

class CustomBackend:
    """Wrapper for custom backend functions"""

    def __init__(self, config: SegmentorConfig):
        self.config = config

    def segment(self, image: PIL_Image, boxes: List[Tuple[int, int, int, int]],
                labels: List[str], scores: Optional[List[float]] = None) -> List[Optional[np.ndarray]]:
        """Use custom backend function if available"""
        if self.config.custom_backend_fn and callable(self.config.custom_backend_fn):
            return self.config.custom_backend_fn(image, boxes, labels, scores)
        return [None] * len(boxes)

def _build_backend(config: SegmentorConfig) -> MaskBackend:
    """Factory function to create the appropriate backend based on configuration"""
    if config.backend == "sam":
        return SamMaskBackend(config)
    elif config.backend == "box_only":
        return BoxOnlyBackend(config)
    elif config.backend == "custom":
        return CustomBackend(config)
    else:
        return BoxOnlyBackend(config)

def segmentor_node(state: GraphState, config: SegmentorConfig = None) -> Dict[str, List[GraspPoint]]:
    """
    Main segmentor node function.

    This function takes detected objects and generates grasp points for each one.
    It supports multiple backends and point selection strategies.

    Args:
        state: GraphState containing image and detections
        config: SegmentorConfig controlling behavior (uses defaults if None)

    Returns:
        Dictionary with 'grasp_points' list
    """
    # Use default config if none provided
    if config is None:
        config = SegmentorConfig()

    # Extract inputs from state
    image = state["image"]
    detections: List[Detection] = state["grounder_output"]

    # Validate inputs
    if not detections:
        return {"grasp_points": []}

    # Process bounding boxes and labels
    boxes: List[Tuple[int, int, int, int]] = [
        _clip_box(tuple(map(int, d["bounding_box"]))) for d in detections
    ]
    labels: List[str] = [str(d["label"]) for d in detections]
    scores = [float(d["score"]) for d in detections] if detections and "score" in detections[0] else None

    # Build backend and generate masks
    backend = _build_backend(config)
    masks = backend.segment(image=image, boxes=boxes, labels=labels, scores=scores)

    # Determine effective points per box
    k_eff = 1 if config.backend == "box_only" else config.points_per_box
    W, H = image.size

    # Generate grasp points
    grasp_points: List[GraspPoint] = []

    for box, label, mask in zip(boxes, labels, masks):
        # Determine point selection mode
        preferred_mode = config.point_mode_map.get(label)
        mode_req = preferred_mode if preferred_mode else config.point_mode

        # Convert mask to 2D if available
        mask_2d = _to_mask2d(mask) if mask is not None else None

        # Auto-infer mode if needed
        if mode_req == "auto":
            chosen_mode = _infer_point_mode(label, box, mask_2d, k_eff)
        else:
            chosen_mode = mode_req

        # Generate points based on chosen mode
        points: List[Tuple[int, int]] = []

        if chosen_mode == "dt":
            if mask_2d is not None:
                points = _mask_to_points_dt(mask_2d, k=k_eff,
                                          suppress_r=config.dt_suppress_radius)
        elif chosen_mode == "centroid":
            if mask_2d is not None:
                c = _centroid_from_mask(mask_2d)
                if c:
                    points = [c]
        elif chosen_mode in {"topmost", "bottommost", "leftmost", "rightmost"}:
            if mask_2d is not None:
                p = _extreme_from_mask(mask_2d, chosen_mode)
                if p:
                    points = [p]
            else:
                # Fallback to bounding box extremes
                x1, y1, x2, y2 = box
                cx, cy = _center(box)
                if chosen_mode == "topmost":
                    points = [(cx, y1)]
                elif chosen_mode == "bottommost":
                    points = [(cx, y2)]
                elif chosen_mode == "leftmost":
                    points = [(x1, cy)]
                elif chosen_mode == "rightmost":
                    points = [(x2, cy)]
        elif chosen_mode == "bbox_center":
            points = [_center(box)]
        elif chosen_mode == "grid":
            points = _grid_points(box, k_eff)

        # Fallback if no points generated
        if not points:
            points = _grid_points(box, k_eff) if k_eff > 1 else [_center(box)]

        # Convert to grasp points and apply coordinate constraints
        for (x, y) in points:
            x = int(max(0, min(W - 1, x)))
            y = int(max(0, min(H - 1, y)))
            u, v = _apply_axis_order((x, y), config.axis_order)

            grasp_point: GraspPoint = {"u": u, "v": v, "label": label}
            grasp_points.append(grasp_point)

    # Log results
    print(f"Segmentor ({config.backend}) generated {len(grasp_points)} grasp points from {len(detections)} detections")

    return {"grasp_points": grasp_points}
