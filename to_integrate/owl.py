import torch
from PIL import Image, ImageDraw
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Dict, Any

class OwlV2Interface:
    """Simplified OWLv2 interface for image files"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"OWLv2 Interface: Using device {self.device}")
        
        model_name = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("OWLv2 Interface: Ready")

    def detect_objects(self, image_path: str, text_queries: List[str], 
                      threshold: float = 0.1, debug: bool = False) -> List[Dict[str, Any]]:
        """
        Detect objects in a saved image file
        
        Args:
            image_path: Path to the image file
            text_queries: List of objects to detect (e.g., ["cat", "dog"])
            threshold: Confidence threshold (0-1)
            debug: Save image with bounding boxes if True
            
        Returns:
            List of detected objects with labels, scores and bounding boxes
        """
        # Load image
        image = Image.open(image_path)
        
        # Format queries with prompt template
        text_labels = [[f"a photo of a {query}" for query in text_queries]]
        
        # Process inputs
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=threshold,
            text_labels=text_labels
        )[0]
        
        # Convert results
        detections = []
        for label, score, box in zip(results["text_labels"], 
                                   results["scores"].cpu().tolist(), 
                                   results["boxes"].cpu().tolist()):
            detections.append({
                "label": label.replace("a photo of a ", ""),
                "score": float(score),
                "box": [float(coord) for coord in box]
            })
        
        # Debug visualization
        if debug:
            draw = ImageDraw.Draw(image)
            for box in results["boxes"].cpu().tolist():
                draw.rectangle(box, outline="red", width=3)
            output_path = image_path.replace(".", "_detected.")
            image.save(output_path)
            print(f"Debug image saved to {output_path}")
            
        return detections


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = OwlV2Interface()
    
    # Detect objects in an image
    results = detector.detect_objects(
        image_path="test.png",  # Your image file
        text_queries=["bottle", "dog", "car", "person"],
        threshold=0.2,
        debug=True
    )
    
    # Print results
    print("\nDetection Results:")
    for obj in results:
        print(f"{obj['label']} ({obj['score']:.2f}): {obj['box']}")
