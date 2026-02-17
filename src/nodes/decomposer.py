from state import GraphState
from langchain_core.messages import SystemMessage, HumanMessage
from llm_config import llm
import os
import logging
import time
from ast import literal_eval

from exceptions.llm_exception import InvalidLLMResponseError

PROMPTS_PATH = "prompts"
MAX_RETRIES = 3
RETRY_DELAY = 1.5  # seconds between retries

def decomposer_node(state: GraphState) -> dict:
    """LLM-based decomposer that transforms high-level instructions into atomic robotic actions, with retry logic."""
    original_prompt = state['original_prompt']
    cwd = os.getcwd()
    path = os.path.join(cwd, PROMPTS_PATH, "decomposer_instruction.txt")

    try:
        with open(path, "r") as f:
            system_prompt = f.read()
    except Exception as e:
        logging.error(f"Failed to read system prompt from {path}: {e}")
        return {
            "decomposed_prompts": []
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=original_prompt)
            ])

            # Optional: Debug print
            print(f"[Attempt {attempt}] LLM response:\n{response.content}")


            decomposed = literal_eval(response.content.strip())  # Safer
            if isinstance(decomposed, list):
                return {
                    "decomposed_prompts": decomposed,
                    "queue": decomposed,  # Initialize queue with decomposed prompts
                    "initial_decomposition_done": True,  # Set flag
                    "current_prompt": decomposed[0] if decomposed else None # Set current prompt
                }
            else:
                logging.warning(f"[Attempt {attempt}] Response was not a list.")

        except Exception as e:
            logging.warning(f"[Attempt {attempt}] Failed to parse LLM response: {e}")
            time.sleep(RETRY_DELAY)

    # All retries failed
    logging.error("All attempts to decompose prompt failed")
    raise InvalidLLMResponseError("All attempts to decompose prompt failed. Abort...")
