"""
Interactive Knowledge Graph Visualizer
Visualizes the chatbot's knowledge graph with interactive 3D and 2D views
Uses networkx, plotly, and pyvis for beautiful visualizations
"""

import json
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("networkx not available. Install with: pip install networkx")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("plotly not available. Install with: pip install plotly")

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("pyvis not available. Install with: pip install pyvis")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not available. Install with: pip install matplotlib")

from knowledge_store import KnowledgeGraphManager, KnowledgeNode, KnowledgeEdge


class KnowledgeGraphVisualizer:
    """Visualize knowledge graph with multiple rendering options"""

    def __init__(self, graph_manager: KnowledgeGraphManager):
        """
        Initialize visualizer

        Args:
            graph_manager: KnowledgeGraphManager instance
        """
        self.graph_manager = graph_manager
        self.nx_graph = None
        self._build_networkx_graph()

    def _build_networkx_graph(self):
        """Build networkx graph from knowledge graph"""
        if not NETWORKX_AVAILABLE:
            return

        self.nx_graph = nx.DiGraph()

        # Get all nodes and edges
        nodes = self.graph_manager.get_all_nodes()
        edges = self.graph_manager.get_all_edges()

        # Add nodes
        for node in nodes:
            self.nx_graph.add_node(
                node.node_id,
                label=node.content[:50],  # Truncate long content
                content=node.content,
                type=node.node_type,
                confidence=node.confidence,
                created_at=node.created_at
            )

        # Add edges
        for edge in edges:
            self.nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation_type,
                weight=edge.weight,
                created_at=edge.created_at
            )

    def refresh(self):
        """Refresh the graph from database"""
        self._build_networkx_graph()

    def get_statistics(self) -> Dict[str, any]:
        """Get graph statistics"""
        if not self.nx_graph:
            return {}

        stats = {
            'total_nodes': self.nx_graph.number_of_nodes(),
            'total_edges': self.nx_graph.number_of_edges(),
            'density': nx.density(self.nx_graph),
            'is_connected': nx.is_weakly_connected(self.nx_graph),
        }

        # Node type distribution
        node_types = defaultdict(int)
        for node, data in self.nx_graph.nodes(data=True):
            node_types[data.get('type', 'unknown')] += 1
        stats['node_types'] = dict(node_types)

        # Edge type distribution
        edge_types = defaultdict(int)
        for source, target, data in self.nx_graph.edges(data=True):
            edge_types[data.get('relation', 'unknown')] += 1
        stats['edge_types'] = dict(edge_types)

        # Centrality (top 5 nodes)
        if self.nx_graph.number_of_nodes() > 0:
            try:
                centrality = nx.degree_centrality(self.nx_graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                stats['top_central_nodes'] = [
                    (node_id, self.nx_graph.nodes[node_id].get('label', node_id), score)
                    for node_id, score in top_nodes
                ]
            except:
                stats['top_central_nodes'] = []

        return stats

    def visualize_2d_static(self, output_path: str = "knowledge_graph_2d.png", layout="spring"):
        """
        Create static 2D visualization

        Args:
            output_path: Output file path
            layout: Layout algorithm (spring, circular, kamada_kawai, etc.)
        """
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
            print("matplotlib and networkx required!")
            return

        # Select layout
        if layout == "spring":
            pos = nx.spring_layout(self.nx_graph, k=0.5, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.nx_graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.nx_graph)
        else:
            pos = nx.spring_layout(self.nx_graph)

        # Create figure
        plt.figure(figsize=(16, 12))

        # Color by node type
        node_types = set(data.get('type', 'unknown') for _, data in self.nx_graph.nodes(data=True))
        color_map = {node_type: i for i, node_type in enumerate(node_types)}
        colors = [color_map[self.nx_graph.nodes[node].get('type', 'unknown')] for node in self.nx_graph.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.nx_graph, pos,
            node_color=colors,
            node_size=500,
            alpha=0.8,
            cmap=plt.cm.Set3
        )

        # Draw edges
        nx.draw_networkx_edges(
            self.nx_graph, pos,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            width=2
        )

        # Draw labels
        labels = {node: data.get('label', str(node))[:20] for node, data in self.nx_graph.nodes(data=True)}
        nx.draw_networkx_labels(self.nx_graph, pos, labels, font_size=8)

        # Create legend
        legend_elements = [
            mpatches.Patch(color=plt.cm.Set3(color_map[node_type]), label=node_type)
            for node_type in node_types
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        plt.title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 2D visualization saved to {output_path}")

    def visualize_2d_interactive(self, output_path: str = "knowledge_graph_interactive.html"):
        """
        Create interactive 2D visualization using plotly

        Args:
            output_path: Output HTML file path
        """
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            print("plotly and networkx required!")
            return

        # Layout
        pos = nx.spring_layout(self.nx_graph, k=0.5, iterations=50)

        # Create edge traces
        edge_traces = []
        for edge in self.nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.nx_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node info
            node_data = self.nx_graph.nodes[node]
            node_text.append(f"{node_data.get('label', node)}<br>Type: {node_data.get('type', 'unknown')}<br>Confidence: {node_data.get('confidence', 0):.2f}")

            # Color by type
            node_type = node_data.get('type', 'unknown')
            node_color.append(hash(node_type) % 20)

            # Size by degree
            node_size.append(10 + self.nx_graph.degree(node) * 5)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[self.nx_graph.nodes[node].get('label', str(node))[:10] for node in self.nx_graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=2, color='white'),
                showscale=True,
                colorbar=dict(
                    title="Node Type",
                    thickness=15,
                    len=0.5
                )
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title="Interactive Knowledge Graph Visualization",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800
        )

        fig.write_html(output_path)
        print(f"✅ Interactive visualization saved to {output_path}")

    def visualize_3d_interactive(self, output_path: str = "knowledge_graph_3d.html"):
        """
        Create interactive 3D visualization using plotly

        Args:
            output_path: Output HTML file path
        """
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            print("plotly and networkx required!")
            return

        # 3D layout
        try:
            pos = nx.spring_layout(self.nx_graph, dim=3, k=0.5, iterations=50)
        except:
            print("⚠️ 3D layout failed, using 2D")
            pos = nx.spring_layout(self.nx_graph, k=0.5, iterations=50)
            # Add z=0 for all nodes
            pos = {node: (*coords, 0) for node, coords in pos.items()}

        # Create edge traces
        edge_traces = []
        for edge in self.nx_graph.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]

            edge_trace = go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_color = []
        node_size = []

        for node in self.nx_graph.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            # Node info
            node_data = self.nx_graph.nodes[node]
            node_text.append(f"{node_data.get('label', node)}<br>Type: {node_data.get('type', 'unknown')}<br>Confidence: {node_data.get('confidence', 0):.2f}")

            # Color by type
            node_type = node_data.get('type', 'unknown')
            node_color.append(hash(node_type) % 20)

            # Size by degree
            node_size.append(10 + self.nx_graph.degree(node) * 5)

        node_trace = go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode='markers+text',
            text=[self.nx_graph.nodes[node].get('label', str(node))[:10] for node in self.nx_graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                colorscale='Viridis',
                line=dict(width=1, color='white'),
                showscale=True,
                colorbar=dict(
                    title="Node Type",
                    thickness=15,
                    len=0.5
                )
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title="Interactive 3D Knowledge Graph",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            width=1200,
            height=800
        )

        fig.write_html(output_path)
        print(f"✅ 3D visualization saved to {output_path}")

    def visualize_pyvis(self, output_path: str = "knowledge_graph_pyvis.html", physics_enabled=True):
        """
        Create interactive visualization using pyvis (physics-based)

        Args:
            output_path: Output HTML file path
            physics_enabled: Enable physics simulation
        """
        if not PYVIS_AVAILABLE:
            print("pyvis required!")
            return

        # Create pyvis network
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=True
        )

        # Enable physics
        if physics_enabled:
            net.toggle_physics(True)
            net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"iterations": 150}
                }
            }
            """)

        # Add nodes
        for node, data in self.nx_graph.nodes(data=True):
            net.add_node(
                node,
                label=data.get('label', str(node))[:30],
                title=f"{data.get('content', '')}",
                color=self._get_color_by_type(data.get('type', 'unknown')),
                size=20 + self.nx_graph.degree(node) * 5
            )

        # Add edges
        for source, target, data in self.nx_graph.edges(data=True):
            net.add_edge(
                source,
                target,
                title=data.get('relation', ''),
                label=data.get('relation', '')[:20],
                arrows='to'
            )

        # Save
        net.save_graph(output_path)
        print(f"✅ Pyvis visualization saved to {output_path}")

    def _get_color_by_type(self, node_type: str) -> str:
        """Get color for node type"""
        colors = {
            'fact': '#4CAF50',
            'concept': '#2196F3',
            'entity': '#FF9800',
            'event': '#9C27B0',
            'unknown': '#757575'
        }
        return colors.get(node_type, '#757575')

    def export_statistics(self, output_path: str = "graph_statistics.json"):
        """Export graph statistics to JSON"""
        stats = self.get_statistics()

        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Statistics exported to {output_path}")

    def find_clusters(self, min_cluster_size=3) -> List[Set[str]]:
        """Find communities/clusters in the graph"""
        if not NETWORKX_AVAILABLE:
            return []

        try:
            # Convert to undirected for community detection
            undirected = self.nx_graph.to_undirected()

            # Use greedy modularity communities
            from networkx.algorithms import community
            communities = community.greedy_modularity_communities(undirected)

            # Filter by size
            clusters = [c for c in communities if len(c) >= min_cluster_size]

            return clusters

        except Exception as e:
            print(f"⚠️ Clustering error: {e}")
            return []


def demo_visualizer():
    """Demo the knowledge graph visualizer"""
    from knowledge_store import VectorStore

    print("Knowledge Graph Visualizer Demo")
    print("=" * 60)

    # Create sample graph
    vector_store = VectorStore("demo_graph.db")
    graph_manager = KnowledgeGraphManager(vector_store)

    # Add sample nodes
    print("Creating sample knowledge graph...")
    nodes = [
        ("Python is a programming language", "fact", 0.9),
        ("Machine learning uses algorithms", "fact", 0.85),
        ("Neural networks are part of ML", "fact", 0.88),
        ("Data science involves analysis", "fact", 0.82),
    ]

    node_ids = []
    for content, node_type, conf in nodes:
        node_id = graph_manager.add_node(content, node_type, confidence=conf)
        node_ids.append(node_id)

    # Add sample edges
    graph_manager.add_edge(node_ids[0], node_ids[1], "used_in")
    graph_manager.add_edge(node_ids[1], node_ids[2], "includes")
    graph_manager.add_edge(node_ids[3], node_ids[1], "uses")

    # Create visualizer
    visualizer = KnowledgeGraphVisualizer(graph_manager)

    # Get statistics
    print("\n📊 Graph Statistics:")
    stats = visualizer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Create visualizations
    print("\n🎨 Creating visualizations...")

    if MATPLOTLIB_AVAILABLE:
        visualizer.visualize_2d_static("demo_graph_2d.png")

    if PLOTLY_AVAILABLE:
        visualizer.visualize_2d_interactive("demo_graph_2d_interactive.html")
        visualizer.visualize_3d_interactive("demo_graph_3d.html")

    if PYVIS_AVAILABLE:
        visualizer.visualize_pyvis("demo_graph_pyvis.html")

    print("\n✅ Demo complete! Check the generated files.")


if __name__ == '__main__':
    demo_visualizer()
