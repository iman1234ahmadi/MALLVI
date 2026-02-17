import torch
import numpy as np
from groundingdino.util.inference import Model as GroundingDINO
from segment_anything import SamPredictor

class GroundedSAM:
    def __init__(self, grounding_dino_config, grounding_dino_checkpoint, sam_checkpoint):
        # Initialize GroundingDINO
        self.grounding_dino = GroundingDINO(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint
        )
        
        # Initialize SAM
        self.sam_predictor = SamPredictor(
            sam_checkpoint=sam_checkpoint
        )
    
    def run(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Process image with GroundingDINO to get boxes, then segment with SAM
        
        Args:
            image: PIL.Image or np.ndarray
            text_prompt: str (e.g., "chair . table . person")
            box_threshold: detection confidence threshold
            text_threshold: text similarity threshold
        """
        # Detect objects with GroundingDINO
        detections = self.grounding_dino.predict_with_caption(
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # Get boxes in [x1, y1, x2, y2] format
        boxes = detections.xyxy  # from GroundingDINO output
        
        # Set image in SAM predictor
        self.sam_predictor.set_image(image)
        
        # Transform boxes to SAM format
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.sam_predictor.device), 
            image.shape[:2]
        )
        
        # Predict masks with SAM
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        return masks, boxes  # Return both masks and original boxes
        
        
        
# Initialize
grounded_sam = GroundedSAM(
    grounding_dino_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="weights/groundingdino_swint_ogc.pth",
    sam_checkpoint="weights/sam_vit_h_4b8939.pth"
)

# Run inference
image = load_your_image()  # PIL Image or numpy array
text_prompt = "chair . table . person"  # Separate classes with dots
masks, boxes = grounded_sam.run(image, text_prompt)

# masks: List of binary masks for each detected object
# boxes: Original bounding boxes from GroundingDINO        
