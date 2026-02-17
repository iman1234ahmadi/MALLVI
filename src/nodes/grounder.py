from state import GraphState, Detection
from typing import Dict, List, Tuple
from PIL import Image
from config.config_classes import GrounderConfig

# Check for optional dependencies
try:
    import torch
    from transformers import (
        Owlv2Processor, Owlv2ForObjectDetection,
        AutoProcessor, AutoModelForZeroShotObjectDetection,
    )
    _HAS_HF = True
except Exception:
    _HAS_HF = False
    torch = None
    Owlv2Processor = Owlv2ForObjectDetection = None
    AutoProcessor = AutoModelForZeroShotObjectDetection = None

def _expand_label_set(canon_labels: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """Expand canonical labels with synonyms for better detection"""
    synonyms = {
        "cube": ["cube", "block", "dice", "die", "box"],
        # Letter objects: map short forms to full names
        "letter A": ["letter A", "A", "letterA"],
        "letter B": ["letter B", "B", "letterB"],
        "letter C": ["letter C", "C", "letterC"],
        "letter D": ["letter D", "D", "letterD"],
        "letter E": ["letter E", "E", "letterE"],
        "letter F": ["letter F", "F", "letterF"],
        "letter G": ["letter G", "G", "letterG"],
        "letter H": ["letter H", "H", "letterH"],
        "letter I": ["letter I", "I", "letterI"],
        "letter J": ["letter J", "J", "letterJ"],
        "letter K": ["letter K", "K", "letterK"],
        "letter L": ["letter L", "L", "letterL"],
        "letter M": ["letter M", "M", "letterM"],
        "letter N": ["letter N", "N", "letterN"],
        "letter O": ["letter O", "O", "letterO"],
        "letter P": ["letter P", "P", "letterP"],
        "letter Q": ["letter Q", "Q", "letterQ"],
        "letter R": ["letter R", "R", "letterR"],
        "letter S": ["letter S", "S", "letterS"],
        "letter T": ["letter T", "T", "letterT"],
        "letter U": ["letter U", "U", "letterU"],
        "letter V": ["letter V", "V", "letterV"],
        "letter W": ["letter W", "W", "letterW"],
        "letter X": ["letter X", "X", "letterX"],
        "letter Y": ["letter Y", "Y", "letterY"],
        "letter Z": ["letter Z", "Z", "letterZ"],
    }
    alias2canon: Dict[str, str] = {}
    query_aliases: List[str] = []

    for canon in canon_labels:
        alist = synonyms.get(canon, [canon])
        for alias in alist:
            if alias not in alias2canon:
                alias2canon[alias] = canon
                query_aliases.append(alias)

    return query_aliases, alias2canon

def _clip_box(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    """Clip bounding box coordinates to image boundaries"""
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union between two bounding boxes"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0

def _fallback_center_box(w: int, h: int, fraction: float) -> Tuple[int, int, int, int]:
    """Generate a fallback bounding box in the center of the image"""
    fw, fh = int(w * fraction), int(h * fraction)
    cx, cy = w // 2, h // 2

    x1, y1 = max(0, cx - fw // 2), max(0, cy - fh // 2)
    x2, y2 = min(w - 1, x1 + fw), min(h - 1, y1 + fh)

    return (x1, y1, x2, y2)

def _accept_by_geometry(canon_label: str, box: Tuple[int, int, int, int]) -> bool:
    """Check if bounding box geometry is acceptable for the given label"""
    if canon_label != "cube":
        return True

    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    ar = h / w

    return 0.8 <= ar <= 1.25

def _detect_owl(img: Image.Image, canon_labels: List[str], config: GrounderConfig) -> \
        Dict[str, Tuple[Tuple[int, int, int, int], float]]:
    """Detect objects using OWLv2 model"""
    if not _HAS_HF:
        return {}

    dev = config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

    # Load model from custom path or HuggingFace
    if config.owl_model_path:
        processor = Owlv2Processor.from_pretrained(config.owl_model_path)
        model = Owlv2ForObjectDetection.from_pretrained(config.owl_model_path).to(dev)
    else:
        processor = Owlv2Processor.from_pretrained(config.owl_model_id)
        model = Owlv2ForObjectDetection.from_pretrained(config.owl_model_id).to(dev)

    query_aliases, alias2canon = _expand_label_set(canon_labels)

    # Format text as simple list (not list of lists) as per user's specification
    try:
        # Create text list: ["a tiles letter T", "a purple frame"]
        texts = [f"a {alias}" for alias in query_aliases]

        # Log input texts for debugging
        print(f"🔍 OWLv2 input texts: {texts}")
        print(f"🔍 OWLv2 query aliases: {query_aliases}")

        # Process inputs with user's exact format
        inputs = processor(text=texts, images=img, return_tensors="pt").to(dev)

        with torch.no_grad():
            outputs = model(**inputs)

    except Exception as e:
        print(f"❌ OWLv2 error: {e}")
        return {}

    # Smart adaptive threshold algorithm for OWLv2 (target exactly 2 objects)
    original_threshold = config.owl_threshold
    # Override with user's specified starting threshold
    if original_threshold > 0.1:
        current_threshold = max(0.1, original_threshold)  # Start from 0.1 if config is higher
    else:
        current_threshold = original_threshold

    min_threshold = 0.01   # Very low minimum for flexibility
    max_threshold = 0.95   # High maximum for precision
    reduction_factor = 0.8  # Reduce by 20% when too few detections
    increase_factor = 1.2   # Increase by 20% when too many detections

    best: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}
    attempt = 1
    max_attempts = 20  # Prevent infinite loops

    while attempt <= max_attempts:
        print(f"🔍 OWLv2 attempt {attempt} with threshold={current_threshold:.3f}")

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([img.size[::-1]]).to(dev)

        # Convert outputs to Pascal VOC Format using user's exact post-processing
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=current_threshold
        )

        # Process results for the first (and only) image
        boxes = results[0]["boxes"].detach().cpu().tolist()
        scores = results[0]["scores"].detach().cpu().tolist()
        labels = results[0]["labels"].detach().cpu().tolist()  # Integer labels corresponding to texts indices

        print(f"🔍 OWLv2 detections: {len(boxes)} (target: exactly 2)")

        W, H = img.size
        current_detections: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}

        # Process all detections without manual filtering
        for b, s, label_idx in zip(boxes, scores, labels):
            # Convert integer label to corresponding text from texts list
            if 0 <= label_idx < len(texts):
                label_text = texts[label_idx]
                # Remove "a " prefix to get the alias
                alias = label_text.replace("a ", "", 1)
                canon = alias2canon.get(alias)
            else:
                continue

            if canon is None:
                continue

            cb = _clip_box(tuple(b), W, H)
            if not _accept_by_geometry(canon, cb):
                continue

            if (canon not in current_detections) or (s > current_detections[canon][1]):
                current_detections[canon] = (cb, float(s))

        num_detections = len(current_detections)

        # Check if we have exactly 2 detections
        if num_detections == 2:
            print(f"✅ OWLv2 found exactly 2 detections using threshold {current_threshold:.3f}")
            return current_detections
        elif num_detections < 2:
            # Too few detections - reduce threshold
            if current_threshold > min_threshold:
                new_threshold = current_threshold * reduction_factor
                current_threshold = max(min_threshold, new_threshold)
                print(f"⚠️  OWLv2 found {num_detections} detections (< 2), "
                      f"reducing threshold to {current_threshold:.3f}")
            else:
                print(f"⚠️  OWLv2 reached minimum threshold {min_threshold:.3f} with {num_detections} detections")
                return current_detections  # Return what we have
        else:
            # Too many detections - increase threshold
            if current_threshold < max_threshold:
                new_threshold = current_threshold * increase_factor
                current_threshold = min(max_threshold, new_threshold)
                print(f"⚠️  OWLv2 found {num_detections} detections (> 2), "
                      f"increasing threshold to {current_threshold:.3f}")
            else:
                print(f"⚠️  OWLv2 reached maximum threshold {max_threshold:.3f} with {num_detections} detections")
                return current_detections  # Return what we have

        attempt += 1

    print(f"⚠️  OWLv2 reached maximum attempts ({max_attempts}) with {len(best)} detections")
    return best

def _dino_postprocess_compat(processor, outputs, inputs, image_size, box_thr, text_thr):
    """Compatibility wrapper for different GroundingDINO post-processing APIs"""
    tried = []

    try:
        return processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids,
            box_threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[image_size[::-1]],
        )[0]
    except TypeError as e:
        tried.append(("post_process_grounded_object_detection(box_threshold, text_threshold)", e))

    try:
        return processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids,
            threshold=box_thr,
            target_sizes=[image_size[::-1]],
        )[0]
    except TypeError as e:
        tried.append(("post_process_grounded_object_detection(threshold)", e))

    if hasattr(processor, "post_process_grounding"):
        try:
            return processor.post_process_grounding(
                outputs=outputs,
                input_ids=inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids,
                box_threshold=box_thr,
                text_threshold=text_thr,
                target_sizes=[image_size[::-1]],
            )[0]
        except TypeError as e:
            tried.append(("post_process_grounding", e))

    msgs = "\n".join([f"- tried {name}: {err}" for name, err in tried])
    raise RuntimeError(f"[GroundingDINO] Incompatible post-process API.\n{msgs}")

def _detect_dino(img: Image.Image, canon_labels: List[str], config: GrounderConfig) -> \
        Dict[str, Tuple[Tuple[int, int, int, int], float]]:
    """Detect objects using GroundingDINO model"""
    if not _HAS_HF:
        return {}

    dev = config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")

    # Load model from custom path or HuggingFace
    if config.dino_model_path:
        processor = AutoProcessor.from_pretrained(config.dino_model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(config.dino_model_path).to(dev)
    else:
        processor = AutoProcessor.from_pretrained(config.dino_model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(config.dino_model_id).to(dev)

    query_aliases, alias2canon = _expand_label_set(canon_labels)

    # Format text as single string with ". " separation as per user's specification
    try:
        # Create text with user's exact format: "a tiles letter T. a purple frame"
        text = ". ".join([f"{alias}" for alias in query_aliases])
        print(f"🔍 GroundingDINO text: {text}")
        print(f"🔍 GroundingDINO query aliases: {query_aliases}")

        # Process inputs with user's exact format
        inputs = processor(images=img, text=text, return_tensors="pt").to(dev)

        with torch.no_grad():
            outputs = model(**inputs)

    except Exception as e:
        print(f"❌ GroundingDINO error: {e}")
        return {}

    # Smart adaptive threshold algorithm for GroundingDINO (target exactly 2 objects)
    original_box_threshold = config.dino_threshold
    original_text_threshold = config.dino_text_threshold
    # Override with user's specified starting thresholds
    current_box_threshold = max(0.3, original_box_threshold)  # Start from 0.3 if config is lower
    current_text_threshold = original_text_threshold

    min_box_threshold = 0.01   # Very low minimum for flexibility
    min_text_threshold = 0.01  # Very low minimum for flexibility
    max_box_threshold = 0.95   # High maximum for precision
    max_text_threshold = 0.95  # High maximum for precision

    box_reduction_factor = 0.9   # Reduce box threshold by 15% when too few detections
    text_reduction_factor = 0.9   # Reduce text threshold by 10% when too few detections
    box_increase_factor = 1.1    # Increase box threshold by 15% when too many detections
    text_increase_factor = 1.1    # Increase text threshold by 10% when too many detections

    best: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}
    attempt = 1
    max_attempts = 20  # Prevent infinite loops

    while attempt <= max_attempts:
        print(f"🔍 GroundingDINO attempt {attempt} with "
              f"box_threshold={current_box_threshold:.3f}, "
              f"text_threshold={current_text_threshold:.3f}")

        # Use user's exact post-processing format
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=current_text_threshold,
            threshold=current_box_threshold,
            target_sizes=[img.size[::-1]]
        )

        # Process results for the first (and only) image
        boxes = results[0]["boxes"].detach().cpu().tolist()
        scores = results[0]["scores"].detach().cpu().tolist()
        labels = results[0]["text_labels"]  # String labels directly

        print(f"🔍 GroundingDINO detections: {len(boxes)} (target: exactly 2)")

        W, H = img.size
        current_detections: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}

        # Process all detections without manual filtering
        for b, s, lab in zip(boxes, scores, labels):
            # Labels are strings directly (not integers like OWL)
            alias = str(lab).strip()
            canon = alias2canon.get(alias)
            if canon is None:
                continue

            cb = _clip_box(tuple(b), W, H)
            if not _accept_by_geometry(canon, cb):
                continue

            if (canon not in current_detections) or (s > current_detections[canon][1]):
                current_detections[canon] = (cb, float(s))

        num_detections = len(current_detections)

        # Check if we have exactly 2 detections
        if num_detections == 2:
            print(f"✅ GroundingDINO found exactly 2 detections using thresholds "
                  f"(box={current_box_threshold:.3f}, text={current_text_threshold:.3f})")
            return current_detections
        elif num_detections < 2:
            # Too few detections - reduce thresholds
            reduced_any = False

            if current_box_threshold > min_box_threshold:
                new_box_threshold = current_box_threshold * box_reduction_factor
                current_box_threshold = max(min_box_threshold, new_box_threshold)
                reduced_any = True

            if current_text_threshold > min_text_threshold:
                new_text_threshold = current_text_threshold * text_reduction_factor
                current_text_threshold = max(min_text_threshold, new_text_threshold)
                reduced_any = True

            if reduced_any:
                print(f"⚠️  GroundingDINO found {num_detections} detections (< 2), "
                      f"reducing thresholds to box={current_box_threshold:.3f}, "
                      f"text={current_text_threshold:.3f}")
            else:
                print(f"⚠️  GroundingDINO reached minimum thresholds with {num_detections} detections")
                return current_detections  # Return what we have
        else:
            # Too many detections - increase thresholds
            increased_any = False

            if current_box_threshold < max_box_threshold:
                new_box_threshold = current_box_threshold * box_increase_factor
                current_box_threshold = min(max_box_threshold, new_box_threshold)
                increased_any = True

            if current_text_threshold < max_text_threshold:
                new_text_threshold = current_text_threshold * text_increase_factor
                current_text_threshold = min(max_text_threshold, new_text_threshold)
                increased_any = True

            if increased_any:
                print(f"⚠️  GroundingDINO found {num_detections} detections (> 2), "
                      f"increasing thresholds to box={current_box_threshold:.3f}, "
                      f"text={current_text_threshold:.3f}")
            else:
                print(f"⚠️  GroundingDINO reached maximum thresholds with {num_detections} detections")
                return current_detections  # Return what we have

        attempt += 1

    print(f"⚠️  GroundingDINO reached maximum attempts ({max_attempts}) with {len(best)} detections")
    return best

def grounder_node(state: GraphState, config: GrounderConfig = None) -> Dict[str, List[Detection]]:
    """
    Main grounder node function.

    This function detects objects in images using grounding models (OWLv2, GroundingDINO)
    based on text descriptions and prepares bounding boxes for the segmentor.
    Searches for all objects_of_interest and excludes not_objects_of_interest.

    Args:
        state: GraphState containing image and object descriptions
        config: GrounderConfig controlling behavior (uses defaults if None)

    Returns:
        Dictionary with 'grounder_output' list of detections containing bounding boxes
    """
    # Use default config if none provided
    if config is None:
        config = GrounderConfig()

    # Extract inputs from state
    image: Image.Image = state["image"]

    # Get single objects of interest and not objects of interest (both are strings)
    object_of_interest = state.get("object_of_interest", "")
    not_object_of_interest = state.get("not_object_of_interest", "")

    # Convert to cleaned strings
    def clean_object_string(obj_str):
        """Clean a single object string"""
        if not obj_str or not isinstance(obj_str, str):
            return ""
        return obj_str.strip()

    # Clean the strings
    object_of_interest = clean_object_string(object_of_interest)
    not_object_of_interest = clean_object_string(not_object_of_interest)

    # Create combined list of ALL objects to search for (both source and destination)
    all_objects_to_find = []

    if object_of_interest:
        all_objects_to_find.append(object_of_interest)

    if not_object_of_interest:
        # Only add destination if it's different from source (avoid duplicates for in-place operations)
        if not_object_of_interest != object_of_interest:
            all_objects_to_find.append(not_object_of_interest)

    print("🔍 Grounder received:")
    print(f"   Object of interest (source): '{object_of_interest}'")
    print(f"   Not object of interest (destination): '{not_object_of_interest}'")
    print(f"   All objects to find: {all_objects_to_find}")

    # If no objects to find, return empty results
    if not all_objects_to_find:
        print("⚠️  No objects to find specified, returning empty detections")
        return {"grounder_output": []}

    # Get image dimensions
    W, H = image.size

    # Initialize detection results
    owl_best: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}
    dino_best: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}

    # Run detection based on mode
    mode = config.grounding_mode.lower()
    print(f"🔍 Grounder using mode: {mode}")

    if mode == "owl":
        owl_best = _detect_owl(image, all_objects_to_find, config)
        print(f"🔍 OWLv2 results: {len(owl_best)} detections")
    elif mode == "dino":
        dino_best = _detect_dino(image, all_objects_to_find, config)
        print(f"🔍 GroundingDINO results: {len(dino_best)} detections")
    elif mode == "both":
        owl_best = _detect_owl(image, all_objects_to_find, config)
        print(f"🔍 OWLv2 results: {len(owl_best)} detections")
        dino_best = _detect_dino(image, all_objects_to_find, config)
        print(f"🔍 GroundingDINO results: {len(dino_best)} detections")
    elif mode == "simple":
        print("🔍 Simple mode: using fallback bounding boxes only")
        owl_best = {}
        dino_best = {}
    else:
        raise ValueError(f"Unsupported grounding_mode='{config.grounding_mode}'. "
                        f"Use 'owl' | 'dino' | 'both' | 'simple'.")

    # Fallback: if both models fail, try the other one
    if mode == "both" and not owl_best and not dino_best:
        print("⚠️  Both models failed, trying OWLv2 as fallback...")
        owl_best = _detect_owl(image, all_objects_to_find, config)

        # If still no results and auto-fallback is enabled, switch to simple mode
        if not owl_best and config.auto_fallback_to_simple:
            print("⚠️  Auto-fallback to simple mode enabled, using fallback bounding boxes...")
            mode = "simple"
    elif mode == "both" and not owl_best:
        print("⚠️  OWLv2 failed, using only GroundingDINO results...")
    elif mode == "both" and not dino_best:
        print("⚠️  GroundingDINO failed, using only OWLv2 results...")

    # Process detections and merge results
    detections_internal: List[dict] = []

    for label in all_objects_to_find:
        candidates: List[Tuple[Tuple[int, int, int, int], float, str]] = []

        # Collect candidates from different sources
        if mode in ("owl", "both") and label in owl_best:
            b, s = owl_best[label]
            candidates.append((b, s, "owl"))

        if mode in ("dino", "both") and label in dino_best:
            b, s = dino_best[label]
            candidates.append((b, s, "dino"))

        # Handle no candidates case
        if len(candidates) == 0:
            print(f"⚠️  No detections found for '{label}', using fallback bounding box")
            fb = _fallback_center_box(W, H, config.fallback_fraction)
            detections_internal.append({"bounding_box": fb, "label": label, "score": 0.0})
        elif len(candidates) == 1:
            b, s, _src = candidates[0]
            detections_internal.append({"bounding_box": b, "label": label, "score": float(s)})
        else:
            # Merge multiple candidates
            (b1, s1, _), (b2, s2, _) = candidates[0], candidates[1]
            if _iou(b1, b2) >= config.iou_merge_threshold:
                final_box, final_score = (b1, max(s1, s2)) if s1 >= s2 else (b2, max(s1, s2))
            else:
                final_box, final_score = (b1, s1) if s1 >= s2 else (b2, s2)
            detections_internal.append({"bounding_box": final_box, "label": label, "score": float(final_score)})

    # Convert to output format
    detections_out: List[Detection] = [
        {"bounding_box": d["bounding_box"], "label": d["label"]}
        for d in detections_internal
    ]

    return {"grounder_output": detections_out}
