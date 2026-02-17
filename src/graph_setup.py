from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import decomposer, perceptor, grounder, segmentor, projector, thinker, actor, reflector
from graph_logger import GraphLogger
from datetime import datetime
from config.config_classes import GrounderConfig, SegmentorConfig

ORIGINAL_PROMPT = "Pick up the red block. Place it on the table. Then, pick the blue cube, and put it on my table."

# Initialize graph
graph = StateGraph(GraphState)

# Initialize logger
logger = GraphLogger(
    folder = "logs",  # Folder where logs will be stored
    log_file = f"robotics_pipeline_{datetime.now().strftime(r'%m_%d_%Y, %H_%M_%S')}.log",
    console_log = False
)

# Grounder configuration will be provided externally
grounder_config: Union[GrounderConfig, Dict[str, Any], None] = None

# Segmentor configuration will be provided externally
segmentor_config: Union[SegmentorConfig, Dict[str, Any], None] = None

# Actor configuration will be provided externally
actor_config: Union[Dict[str, Any], None] = None

def set_grounder_config(config: Union[GrounderConfig, Dict[str, Any]]):
    """Set the grounder configuration to be used by the graph.

    Args:
        config: GrounderConfig object or dictionary configuration
    """
    global grounder_config
    grounder_config = config

    # Log configuration details
    if isinstance(config, GrounderConfig):
        config_dict = {
            "grounding_mode": config.grounding_mode,
            "device": config.device,
            "auto_fallback_to_simple": config.auto_fallback_to_simple
        }
    else:
        config_dict = config

    logger.log("Grounder configuration set", {"config": config_dict})
    logger.flush()  # Ensure configuration is written to file

def set_segmentor_config(config: Union[SegmentorConfig, Dict[str, Any]]):
    """Set the segmentor configuration to be used by the graph.

    Args:
        config: SegmentorConfig object or dictionary configuration
    """
    global segmentor_config
    segmentor_config = config

    # Log configuration details
    if isinstance(config, SegmentorConfig):
        config_dict = {
            "backend": config.backend,
            "device": config.device,
            "points_per_box": config.points_per_box
        }
    else:
        config_dict = config

    logger.log("Segmentor configuration set", {"config": config_dict})
    logger.flush()  # Ensure configuration is written to file

def set_actor_config(config: Dict[str, Any]):
    """Set the actor configuration to be used by the graph.

    Args:
        config: Dictionary configuration for actor mode
               {"mode": "oracle" | "pipeline"}
    """
    global actor_config
    actor_config = config

    logger.log("Actor configuration set", {"config": config})
    logger.flush()  # Ensure configuration is written to file



# Create wrapped nodes with enhanced logging
def create_logged_node(original_func, name: str):
    def wrapper(state: Dict):
        # Log initial state
        logger.log(f"Node started: {name}", {"state": logger.summarize_state(state)})

        # Execute the node with configuration if needed
        if name == "grounder":
            # For grounder, use the provided config object directly or create default
            if grounder_config:
                # Use provided configuration (object or dict)
                if isinstance(grounder_config, GrounderConfig):
                    config = grounder_config
                else:
                    # Convert dict to config object
                    config = GrounderConfig(**grounder_config)
            else:
                # Use default configuration with simple mode
                config = GrounderConfig(grounding_mode="simple")
            result = original_func(state, config)
        elif name == "segmentor":
            # For segmentor, use the provided config object directly or create default
            if segmentor_config:
                # Use provided configuration (object or dict)
                if isinstance(segmentor_config, SegmentorConfig):
                    config = segmentor_config
                else:
                    # Convert dict to config object
                    config = SegmentorConfig(**segmentor_config)
            else:
                # Use default configuration with box_only mode
                config = SegmentorConfig(backend="box_only")
            result = original_func(state, config)
        elif name == "actor":
            # For actor, create config from dict or use default
            from nodes.actor import ActorConfig, ActorMode
            if actor_config:
                mode = ActorMode(actor_config.get("mode", "pipeline"))
                config = ActorConfig(mode=mode)
            else:
                # Use default configuration (pipeline mode)
                config = ActorConfig(mode=ActorMode.PIPELINE)
            result = original_func(state, config)
        else:
            result = original_func(state)

        # Log final state
        logger.log(f"Node completed: {name}", {"result": logger.summarize_state(result)})

        return result
    return wrapper

# Add logged nodes to graph
graph.add_node("decomposer", create_logged_node(decomposer.decomposer_node, "decomposer"))
graph.add_node("perceptor", create_logged_node(perceptor.perceptor_node, "perceptor"))
graph.add_node("grounder", create_logged_node(grounder.grounder_node, "grounder"))
graph.add_node("segmentor", create_logged_node(segmentor.segmentor_node, "segmentor"))
graph.add_node("projector", create_logged_node(projector.projector_node, "projector"))
graph.add_node("thinker", create_logged_node(thinker.thinker_node, "thinker"))
graph.add_node("actor", create_logged_node(actor.actor_node, "actor"))
graph.add_node("reflector", create_logged_node(reflector.reflector_node, "reflector"))

# Set entry point
graph.set_entry_point("decomposer")

# Decomposer always goes to perceptor
graph.add_edge("decomposer", "perceptor")

# Perceptor processes the first prompt from the queue
graph.add_edge("perceptor", "grounder")

# Main processing pipeline
graph.add_edge("grounder", "segmentor")
graph.add_edge("segmentor", "projector")
graph.add_edge("projector", "thinker")
graph.add_edge("thinker", "actor")
graph.add_edge("actor", "reflector")

# Conditional edge after reflector
graph.add_conditional_edges(
    "reflector",
    lambda state: "END" if not state["queue"] and state["should_terminate"] else "perceptor",
    {
        "perceptor": "perceptor",
        "END": END,
    }
)

# Compile graph
app = graph.compile()

# Example usage (for testing)
if __name__ == "__main__":
    print("🔧 Graph setup module loaded successfully")
    print("📋 Use set_vima_interface() to configure VIMA interface")
    print("🚀 Use main.py to run the complete pipeline")
