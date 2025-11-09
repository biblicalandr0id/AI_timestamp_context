import matplotlib.pyplot as plt
from ai_consciousness_model import ConsciousnessEngine

class ConsciousnessVisualizer:
    def __init__(self, engine: ConsciousnessEngine):
        self.engine = engine
        
    def plot_consciousness_timeline(self):
        plt.figure(figsize=(15, 10))
        
        # Plot consciousness scores over time
        times = [s.timestamp for s in self.engine.states]
        scores = [s.consciousness_score for s in self.engine.states]
        
        plt.subplot(2, 1, 1)
        plt.plot(times, scores, 'b-', label='Consciousness Score')
        plt.title('Consciousness Evolution Over Time')
        plt.ylabel('Consciousness Score')
        
        # Plot attention patterns
        plt.subplot(2, 1, 2)
        attention_data = [len(s.attention_focus) for s in self.engine.states]
        plt.plot(times, attention_data, 'r-', label='Attention Complexity')
        plt.xlabel('Time')
        plt.ylabel('Attention Focus Points')
        
        plt.tight_layout()
        return plt