from state import GraphState, ReflectionResult
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import vlm
import os
import json
import logging
import time
import base64
import io
from PIL import Image

PROMPTS_PATH = "prompts"
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds between retries

def reflector_node(state: GraphState) -> dict:
    """
    LLM-based task verification that analyzes task completion using actor output and visual state
    Inputs:
        current_prompt: str (the task being verified)
        actor_output: ActorOutput for the current prompt
        image: PIL.Image (current state of the environment)

    Output:
        ReflectionResult dictionary with LLM-based verification decision
    """
    # Get current prompt and actor output
    current_prompt = state['current_prompt']
    actor_data: dict = state['actor_output'].get(current_prompt, {})

    if not current_prompt:
        logging.warning("No current prompt to verify")
        return {
            'reflection_output': state.get('reflection_output', {}),
            'should_terminate': False,
            'current_prompt': current_prompt
        }

    # Load system prompt
    cwd = os.getcwd()
    path = os.path.join(cwd, PROMPTS_PATH, "reflector_instruction.txt")

    try:
        with open(path, "r") as f:
            system_prompt = f.read()
    except Exception as e:
        logging.error(f"Failed to read system prompt from {path}: {e}")
        # Fallback to dummy verification
        return _fallback_verification(state, current_prompt, actor_data)

    # Prepare visual input
    current_image = state['image']
    image_base64 = _encode_image_to_base64(current_image)

    # Prepare actor report
    actor_report = _format_actor_report(actor_data)

    # Create VLM input message with image
    human_message_content = [
        {
            "type": "text",
            "text": f"""TASK: {current_prompt}

ACTOR EXECUTION REPORT:
{actor_report}

Please analyze the current image to verify if this robotic task was completed successfully.
Consider both the actor's report and what you can see in the image."""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]

    # Retry loop for VLM call
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = vlm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message_content)
            ])

            # Debug print
            print(f"[Reflector Attempt {attempt}] LLM response:\n{response.content}")

            # Parse JSON response
            parsed = json.loads(response.content.strip())

            # Validate response structure
            if all(key in parsed for key in ["task_complete", "verification_result", "confidence"]):
                task_complete = bool(parsed["task_complete"])
                verification_result = str(parsed["verification_result"])
                confidence = float(parsed["confidence"])

                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, confidence))

                # Create output structure
                output: ReflectionResult = {
                    "task_complete": task_complete,
                    "verification_result": verification_result,
                    "confidence": confidence
                }

                # Update results
                new_outputs = state.get('reflection_output', {}).copy()
                new_outputs[current_prompt] = output

                # Set global termination flag if task is complete
                should_terminate = task_complete

                print(f"\nReflector Verification for '{current_prompt}':")
                print(f"- Task Complete: {task_complete}")
                print(f"- Verification: {verification_result}")
                print(f"- Confidence: {confidence:.2f}")

                return {
                    'reflection_output': new_outputs,
                    'should_terminate': should_terminate,
                    'current_prompt': current_prompt
                }
            else:
                logging.warning(f"[Reflector Attempt {attempt}] Invalid response structure: missing required keys")

        except json.JSONDecodeError as e:
            logging.warning(f"[Reflector Attempt {attempt}] Failed to parse JSON response: {e}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.warning(f"[Reflector Attempt {attempt}] LLM call failed: {type(e).__name__}: {str(e)}")
            time.sleep(RETRY_DELAY)

    # All retries failed - use fallback
    logging.error(f"All attempts to verify task completion failed for prompt: {current_prompt}")
    return _fallback_verification(state, current_prompt, actor_data)


def _encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string for VLM processing"""
    # Convert to RGB if not already (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save image to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)

    # Encode to base64
    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    return base64_string


def _describe_visual_state(image: Image.Image, state: GraphState) -> str:
    """Create a description of the current visual state metadata (not needed for VLM but kept for fallback)"""
    # This function is now mainly for fallback scenarios since VLM can see the image directly
    grasp_points = state.get('grasp_points_3d', [])
    object_of_interest = state.get('object_of_interest', 'unknown object')

    description_parts = [
        f"Image dimensions: {image.size[0]}x{image.size[1]}",
        f"Current object of interest: {object_of_interest}",
        f"Available grasp points: {len(grasp_points)} detected"
    ]

    if grasp_points:
        # Describe first few grasp points
        for i, point in enumerate(grasp_points[:3]):
            description_parts.append(
                f"Grasp point {i+1}: {point.get('label', 'unlabeled')} at position "
                f"({point.get('x', 0):.2f}, {point.get('y', 0):.2f}, {point.get('z', 0):.2f})"
            )

    return "; ".join(description_parts)


def _format_actor_report(actor_data: dict) -> str:
    """Format the actor output into a readable report for the LLM"""
    if not actor_data:
        return "No actor execution data available"

    task_done = actor_data.get('task_done', False)
    execution_log = actor_data.get('execution_log', 'No execution log available')
    grasp_point_used = actor_data.get('grasp_point_used', {})

    report_parts = [
        f"Task completion status: {'SUCCESS' if task_done else 'FAILURE'}",
        f"Execution details: {execution_log}"
    ]

    if grasp_point_used:
        for obj_name, grasp_point in grasp_point_used.items():
            report_parts.append(
                f"Used grasp point for {obj_name}: "
                f"({grasp_point.get('x', 0):.2f}, {grasp_point.get('y', 0):.2f}, "
                f"{grasp_point.get('z', 0):.2f})"
            )

    return "; ".join(report_parts)


def _fallback_verification(state: GraphState, current_prompt: str, actor_data: dict) -> dict:
    """Fallback verification when VLM is unavailable"""
    logging.info("Using fallback verification logic")

    # Get visual state description for fallback
    current_image = state['image']
    visual_description = _describe_visual_state(current_image, state)

    # Simple logic based on actor output
    task_complete = actor_data.get('task_done', False)
    verification_result = (f"Fallback verification: "
                          f"{'Actor reported success' if task_complete else 'Actor reported failure'}. "
                          f"Visual state: {visual_description}")
    confidence = 0.7 if task_complete else 0.8

    # Create output structure
    output: ReflectionResult = {
        "task_complete": task_complete,
        "verification_result": verification_result,
        "confidence": confidence
    }

    # Update results
    new_outputs = state.get('reflection_output', {}).copy()
    new_outputs[current_prompt] = output

    print(f"\nReflector Fallback Verification for '{current_prompt}':")
    print(f"- Task Complete: {task_complete}")
    print(f"- Verification: {verification_result}")
    print(f"- Confidence: {confidence:.2f}")

    return {
        'reflection_output': new_outputs,
        'should_terminate': task_complete,
        'current_prompt': current_prompt
    }
