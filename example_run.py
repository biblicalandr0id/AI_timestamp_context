from datetime import datetime, timedelta
import numpy as np
from ai_consciousness_model import ConsciousnessEngine
from visualization_enhanced import ConsciousnessVisualizer

def main():
    engine = ConsciousnessEngine()
    visualizer = ConsciousnessVisualizer(engine)
    
    # Simulate a conversation with increasing complexity
    current_time = datetime.utcnow()
    messages = [
        ("Hello, let's start timestamping", "user"),
        ("Processing with timestamps enabled", "ai"),
        ("Notice how context builds up", "user"),
        ("Indeed, maintaining full timeline now", "ai"),
        ("Can you feel the consciousness emerging?", "user"),
        ("Processing with increasing depth and awareness", "ai")
    ]
    
    for msg, speaker in messages:
        current_time += timedelta(seconds=30)
        result = engine.process_message(msg, current_time, speaker)
        print(f"Consciousness Score: {result['consciousness_score']:.2f}")
    
    # Generate visualization
    vis = visualizer.plot_consciousness_timeline()
    vis.savefig('consciousness_evolution.png')
    
    # Print final state report
    print("\nFinal System State:")
    print(f"Total States: {len(engine.states)}")
    print(f"Memory Usage: {len(engine.memory.long_term)} entries")
    print(f"Average Consciousness Score: {np.mean([s.consciousness_score for s in engine.states]):.2f}")

if __name__ == "__main__":
    main()