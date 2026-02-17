from state import GraphState, ThinkerOutput
import json
import logging
from llm_config import llm
import os

PROMPTS_PATH = "prompts"
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds between retries

def load_thinker_prompt() -> str:
    """Load the thinker instruction prompt."""
    cwd = os.getcwd()
    path = os.path.join(cwd, PROMPTS_PATH, "thinker_instruction.txt")

    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to read thinker prompt from {path}: {e}")
        # Fallback prompt
        return """You are an intelligent robotics task planner. Generate a structured action plan for pick-and-place operations.

Output format:
{
  "decision": "SUCCESS" or "FAILURE",
  "chosen_grasp_points": [[[pick_x, pick_y, pick_z], [place_x, place_y, place_z]]],
  "reasoning": "Explanation of decisions",
  "rotation_degrees": [rotation1, rotation2, ...]
}"""

def thinker_node(state: GraphState) -> dict:
    """
    LLM-based intelligent task planner that generates pick-place actions.

    Inputs:
        current_prompt: string (from perceptor queue)
        object_of_interest: string or list of strings (objects to be picked up)
        not_object_of_interest: string (destination object where things get placed)
        grasp_points_3d: list[GraspPoint3D] (from projector - contains ALL objects)

    Output:
        ThinkerOutput dictionary with intelligent action planning
    """
    # Get current prompt and associated data
    current_prompt = state['current_prompt']
    obj_interest = state['object_of_interest']
    not_objects = state['not_object_of_interest']
    grasp_points = state['grasp_points_3d']

    print(f"\n🤖 Thinker LLM Agent processing prompt: '{current_prompt}'")
    print(f"🎯 Object(s) of interest (source): {obj_interest}")
    print(f"📍 Destination objects: {not_objects}")
    print(f"🔍 Available grasp points: {len(grasp_points)}")

    # Load system prompt
    system_prompt = load_thinker_prompt()

    # Prepare input for LLM with clear labeling
    human_message = f"""Please analyze this task and generate an action plan:

PROMPT: {current_prompt}

SOURCE OBJECTS (to be picked up): {obj_interest}
DESTINATION OBJECTS (where things get placed): {not_objects}

AVAILABLE GRASP POINTS FOR ALL OBJECTS: {json.dumps(grasp_points, indent=2)}

IMPORTANT:
- Pick positions should come from grasp points matching the SOURCE OBJECTS
- Place positions should come from grasp points matching the DESTINATION OBJECTS
- For place positions, use the top surface of destination objects (adjust Z coordinate upward)
- Generate pick-place pairs by matching source objects with destination objects
- **Rotation is ALWAYS part of pick-place operations - use 0° when no rotation specified**
- For in-place rotation (same source and destination), pick and place at same position
- Each action must have a rotation value (0° if no rotation needed)

Generate a structured action plan following the specified format."""

    # Retry loop for LLM response
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🔄 LLM attempt {attempt}/{MAX_RETRIES}...")

            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_message}
            ])

            print(f"📝 LLM response received (attempt {attempt})")
            # print(response.content.strip())

            # Parse JSON response
            try:
                parsed = json.loads(response.content.strip())

                # Validate required fields
                required_fields = ["decision", "chosen_grasp_points", "reasoning", "rotation_degrees"]
                if all(field in parsed for field in required_fields):

                    # Validate data types and structure
                    if (isinstance(parsed["decision"], str) and
                        isinstance(parsed["chosen_grasp_points"], list) and
                        isinstance(parsed["reasoning"], str) and
                        isinstance(parsed["rotation_degrees"], list)):

                        print("✅ LLM response parsed successfully")

                        # Create output structure matching ThinkerOutput template
                        output: ThinkerOutput = {
                            "decision": parsed["decision"],
                            "chosen_grasp_points": (parsed["chosen_grasp_points"]
                                                   if parsed["chosen_grasp_points"]
                                                   else None),
                            "reasoning": parsed["reasoning"],
                            "rotation_degrees": parsed["rotation_degrees"]
                        }

                        # Update results
                        new_outputs = state.get('thinker_output', {}).copy()
                        new_outputs[current_prompt] = output

                        print(f"\n🎯 Thinker Decision for '{current_prompt}':")
                        print(f"   • Decision: {output['decision']}")
                        actions_count = len(output['chosen_grasp_points']) if output['chosen_grasp_points'] else 0
                        print(f"   • Actions: {actions_count}")
                        print(f"   • Rotations: {output['rotation_degrees']}")
                        reasoning_preview = output['reasoning'][:100]
                        reasoning_suffix = '...' if len(output['reasoning']) > 100 else ''
                        print(f"   • Reasoning: {reasoning_preview}{reasoning_suffix}")

                        return {
                            'thinker_output': new_outputs
                        }
                    else:
                        print("❌ Invalid data types in LLM response")
                        continue
                else:
                    print("❌ Missing required fields in LLM response")
                    continue

            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON from LLM response: {e}")
                print(f"Raw response: {response.content[:200]}...")
                continue

        except Exception as e:
            print(f"❌ LLM invocation failed (attempt {attempt}): {type(e).__name__}: {str(e)}")
            if attempt < MAX_RETRIES:
                import time
                time.sleep(RETRY_DELAY)
            continue

    # All retries failed - fallback to error state
    print(f"❌ All LLM attempts failed for prompt: {current_prompt}")

    # Create fallback output
    fallback_output: ThinkerOutput = {
        "decision": "FAILURE",
        "chosen_grasp_points": None,
        "reasoning": (f"LLM agent failed to process prompt after {MAX_RETRIES} attempts. "
                     f"Unable to generate action plan."),
        "rotation_degrees": []
    }

    # Update results with fallback
    new_outputs = state.get('thinker_output', {}).copy()
    new_outputs[current_prompt] = fallback_output

    print(f"⚠️  Using fallback output: {fallback_output['decision']}")

    return {
        'thinker_output': new_outputs
    }
