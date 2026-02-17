#!/usr/bin/env python3
"""
Graph Manager
=============

This module handles LangGraph setup and configuration.
It provides a clean interface for creating and managing the robotic manipulation pipeline graph.
"""

from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import decomposer, perceptor, grounder, segmentor, projector, thinker, actor, reflector
from graph_logger import GraphLogger
from datetime import datetime
from config.config_classes import GrounderConfig, SegmentorConfig

from utils.printer import Printer

class GraphManager:
    """
    Manages LangGraph setup and configuration.
    """
    
    def __init__(self, 
                 grounder_config: Union[GrounderConfig, Dict[str, Any], None] = None,
                 segmentor_config: Union[SegmentorConfig, Dict[str, Any], None] = None,
                 actor_config: Union[Dict[str, Any], None] = None,
                 logger: GraphLogger = None):
        """
        Initialize the graph manager.
        
        Args:
            grounder_config: Grounder configuration
            segmentor_config: Segmentor configuration  
            actor_config: Actor configuration
            logger: Graph logger instance
        """
        self.grounder_config = grounder_config
        self.segmentor_config = segmentor_config
        self.actor_config = actor_config
        self.logger = logger or self._create_default_logger()
        
        Printer.debug("Initializing Graph Manager")
        Printer.config("Grounder config", 'Provided' if grounder_config else 'Default')
        Printer.config("Segmentor config", 'Provided' if segmentor_config else 'Default')
        Printer.config("Actor config", 'Provided' if actor_config else 'Default')
        
        # Create the graph
        self.graph = self._create_graph()
        self.app = self.graph.compile()
        
        Printer.success("Graph Manager initialized successfully")
    
    def _create_default_logger(self) -> GraphLogger:
        """Create a default graph logger"""
        return GraphLogger(
            folder="logs",
            log_file=f"robotics_pipeline_{datetime.now().strftime(r'%m_%d_%Y, %H_%M_%S')}.log",
            console_log=False
        )
    
    def _create_logged_node(self, original_func, name: str):
        """Create a logged wrapper for a node function"""
        def wrapper(state: Dict):
            # Log initial state
            self.logger.log(f"Node started: {name}", {"state": self.logger.summarize_state(state)})
            
            # Execute the node with configuration if needed
            if name == "grounder":
                config = self._get_grounder_config()
                result = original_func(state, config)
            elif name == "segmentor":
                config = self._get_segmentor_config()
                result = original_func(state, config)
            elif name == "actor":
                config = self._get_actor_config()
                result = original_func(state, config)
            else:
                result = original_func(state)
            
            # Log final state
            self.logger.log(f"Node completed: {name}", {"result": self.logger.summarize_state(result)})
            
            return result
        return wrapper
    
    def _get_grounder_config(self) -> GrounderConfig:
        """Get grounder configuration"""
        if self.grounder_config:
            if isinstance(self.grounder_config, GrounderConfig):
                return self.grounder_config
            else:
                return GrounderConfig(**self.grounder_config)
        else:
            return GrounderConfig(grounding_mode="simple")
    
    def _get_segmentor_config(self) -> SegmentorConfig:
        """Get segmentor configuration"""
        if self.segmentor_config:
            if isinstance(self.segmentor_config, SegmentorConfig):
                return self.segmentor_config
            else:
                return SegmentorConfig(**self.segmentor_config)
        else:
            return SegmentorConfig(backend="box_only")
    
    def _get_actor_config(self):
        """Get actor configuration"""
        if self.actor_config:
            from nodes.actor import ActorConfig, ActorMode
            mode = ActorMode(self.actor_config.get("mode", "pipeline"))
            return ActorConfig(mode=mode)
        else:
            from nodes.actor import ActorConfig, ActorMode
            return ActorConfig(mode=ActorMode.PIPELINE)
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph with all nodes and edges"""
        Printer.debug("Creating LangGraph structure...")
        
        # Create graph
        graph = StateGraph(GraphState)
        
        # Add logged nodes to graph
        Printer.debug("  - Adding nodes...")
        graph.add_node("decomposer", self._create_logged_node(decomposer.decomposer_node, "decomposer"))
        graph.add_node("perceptor", self._create_logged_node(perceptor.perceptor_node, "perceptor"))
        graph.add_node("grounder", self._create_logged_node(grounder.grounder_node, "grounder"))
        graph.add_node("segmentor", self._create_logged_node(segmentor.segmentor_node, "segmentor"))
        graph.add_node("projector", self._create_logged_node(projector.projector_node, "projector"))
        graph.add_node("thinker", self._create_logged_node(thinker.thinker_node, "thinker"))
        graph.add_node("actor", self._create_logged_node(actor.actor_node, "actor"))
        graph.add_node("reflector", self._create_logged_node(reflector.reflector_node, "reflector"))
        
        # Set entry point
        graph.set_entry_point("decomposer")
        
        # Add edges
        Printer.debug("  - Adding edges...")
        graph.add_edge("decomposer", "perceptor")
        graph.add_edge("perceptor", "grounder")
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
        
        Printer.success("LangGraph structure created successfully")
        return graph
    
    def get_app(self):
        """Get the compiled graph application"""
        return self.app
    
    def get_logger(self) -> GraphLogger:
        """Get the graph logger"""
        return self.logger
    
    def update_configs(self, 
                      grounder_config: Union[GrounderConfig, Dict[str, Any], None] = None,
                      segmentor_config: Union[SegmentorConfig, Dict[str, Any], None] = None,
                      actor_config: Union[Dict[str, Any], None] = None):
        """
        Update node configurations.
        
        Args:
            grounder_config: New grounder configuration
            segmentor_config: New segmentor configuration
            actor_config: New actor configuration
        """
        if grounder_config is not None:
            self.grounder_config = grounder_config
            Printer.debug("Grounder configuration updated")
        
        if segmentor_config is not None:
            self.segmentor_config = segmentor_config
            Printer.debug("Segmentor configuration updated")
        
        if actor_config is not None:
            self.actor_config = actor_config
            Printer.debug("Actor configuration updated")
    
    def log_configuration(self):
        """Log current configuration details"""
        self.logger.log("Graph configuration", {
            "grounder_config": self._get_grounder_config().__dict__ if self.grounder_config else "default",
            "segmentor_config": self._get_segmentor_config().__dict__ if self.segmentor_config else "default",
            "actor_config": self.actor_config or "default"
        })
        self.logger.flush()

# Convenience function for backward compatibility
def create_graph_manager(grounder_config=None, segmentor_config=None, actor_config=None, logger=None) -> GraphManager:
    """
    Create a graph manager instance.
    
    Args:
        grounder_config: Grounder configuration
        segmentor_config: Segmentor configuration
        actor_config: Actor configuration
        logger: Graph logger instance
        
    Returns:
        GraphManager instance
    """
    return GraphManager(grounder_config, segmentor_config, actor_config, logger)
