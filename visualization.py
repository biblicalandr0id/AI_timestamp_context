import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta

class ConversationVisualizer:
    def __init__(self, conversation_system: EnhancedConversationSystem):
        self.system = conversation_system
        
    def plot_timeline(self):
        plt.figure(figsize=(15, 8))
        G = self.system.conversation_graph
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Draw nodes and edges
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=1000,
                with_labels=True,
                labels={node: f"{str(node.time())}" for node in G.nodes()})
        
        plt.title("Conversation Flow Timeline")
        return plt

    def generate_context_report(self) -> str:
        report = []
        for state in self.system.states:
            report.append(
                f"[{state.timestamp}] {state.user}\n"
                f"Context Depth: {state.context_depth}\n"
                f"Patterns: {', '.join(state.patterns or [])}\n"
                f"Content: {state.content}\n"
                f"-" * 50
            )
        return "\n".join(report)