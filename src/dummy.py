from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the State structure


class MessageState(TypedDict):
    message: str

# Define node functions


def node_a(state: MessageState) -> dict:
    print("Node A processing...")
    return {"message": state["message"] + " A"}


def node_b(state: MessageState) -> dict:
    print("Node B processing...")
    return {"message": state["message"] + " B"}


def node_c(state: MessageState) -> dict:
    print("Node C processing...")
    return {"message": state["message"] + " C"}


# Build the graph


graph = StateGraph(MessageState)

# Add nodes
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

# Set entry point
graph.set_entry_point("node_a")

# Define edges
graph.add_edge("node_a", "node_b")
graph.add_edge("node_b", "node_c")
graph.add_edge("node_c", END)  # Connect to END

# Compile the graph
app = graph.compile()

# Execute the graph
initial_state = {"message": "Start"}
result = app.invoke(initial_state)

print("\nFinal Result:", result["message"])
