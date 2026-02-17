from state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import llm
import os
import json
import logging
import time

from exceptions.llm_exception import InvalidLLMResponseError

PROMPTS_PATH = "prompts"
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds between retries

def perceptor_node(state: GraphState) -> dict:
    """LLM-based entity extractor that identifies objects of interest from atomic prompts"""
    # Handle initial decomposition - check if we need to initialize queue
    initial_decomposition_done = state.get('initial_decomposition_done', False)
    decomposed_prompts = state.get('decomposed_prompts', [])
    existing_queue = state.get('queue', [])

    if not initial_decomposition_done and decomposed_prompts and not existing_queue:
        # First run - initialize queue from decomposed_prompts
        queue = decomposed_prompts.copy()
        initial_decomposition_done = True
        print(f"📋 Initialized queue with {len(queue)} decomposed prompts")
    else:
        queue = existing_queue.copy()

    if not queue:
        # Nothing to process
        return {
            'queue': [],
            'current_prompt': None,
            'object_of_interest': "",
            'not_object_of_interest': "",
            'results': state.get('results', {}),
            'initial_decomposition_done': initial_decomposition_done
        }

    # Initialize results safely
    results = state.get('results', {}).copy()
    current_prompt = state.get('current_prompt', None)
    task_complete = state.get('reflection_output', {}).get(current_prompt, {}).get('task_complete', False)
    if task_complete or initial_decomposition_done:
        state['initial_decomposition_done'] = False
        current_prompt = queue.pop(0)

    # Load system prompt
    cwd = os.getcwd()
    path = os.path.join(cwd, PROMPTS_PATH, "perceptor_instruction.txt")

    try:
        with open(path, "r") as f:
            system_prompt = f.read()
    except Exception as e:
        logging.error(f"Failed to read system prompt from {path}: {e}")
        return {
            'queue': queue,
            'current_prompt': current_prompt,
            'object_of_interest': "",
            'not_object_of_interest': "",
            'results': results,
            'initial_decomposition_done': initial_decomposition_done
        }

    # Retry loop
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=current_prompt)
            ])

            # Debug print
            print(f"[Perceptor Attempt {attempt}] LLM response:\n{response.content}")

            # PARSE SAFELY WITH JSON
            parsed = json.loads(response.content.strip())
            object_of_interest = parsed.get("object_of_interest", "")
            not_object = parsed.get("not_object", "")

            # Validate and handle types - keep as single strings for grounder compatibility
            if isinstance(not_object, str):
                # Single object string - use as is
                not_object_of_interest = not_object.strip()
            else:
                # If it's not a string, convert to string
                not_object_of_interest = str(not_object).strip()

            # Handle object_of_interest - should be a single string
            if isinstance(object_of_interest, str):
                # Single object string - use as is
                final_object_of_interest = object_of_interest.strip()
            elif isinstance(object_of_interest, list) and len(object_of_interest) > 0:
                # If it's a list, take the first item (should typically be single item)
                final_object_of_interest = str(object_of_interest[0]).strip()
            else:
                # Invalid or empty
                logging.warning(f"[Perceptor Attempt {attempt}] Invalid object_of_interest: {object_of_interest}")
                continue

            # Store results (keep original format for results, but return strings for grounder)
            results[current_prompt] = {
                'object_of_interest': final_object_of_interest,
                'not_object_of_interest': not_object_of_interest
            }

            return {
                'queue': queue,
                'current_prompt': current_prompt,
                'object_of_interest': final_object_of_interest,  # Single string
                'not_object_of_interest': not_object_of_interest,  # Single string
                'results': results
            }

        except Exception as e:
            logging.warning(f"[Perceptor Attempt {attempt}] Failed to parse LLM response: {type(e).__name__}: {str(e)}")
            time.sleep(RETRY_DELAY)

    # All retries failed
    logging.error(f"All attempts to extract entities from prompt failed: {current_prompt}")
    raise InvalidLLMResponseError("Perceptor failed to extract entities. Aborting...")
