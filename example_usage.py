from datetime import datetime, timedelta
from conversation_system import EnhancedConversationSystem
from visualization import ConversationVisualizer

def main():
    system = EnhancedConversationSystem()
    visualizer = ConversationVisualizer(system)
    
    # Simulate conversation
    current_time = datetime.utcnow()
    messages = [
        ("Hello, let's start timestamping", "user"),
        ("Processing with timestamps enabled", "ai"),
        ("Notice how context builds up", "user"),
        ("Indeed, maintaining full timeline now", "ai")
    ]
    
    for msg, speaker in messages:
        current_time += timedelta(seconds=30)
        system.process_message(msg, current_time, speaker)
    
    # Generate visualization
    vis = visualizer.plot_timeline()
    vis.savefig('conversation_timeline.png')
    
    # Print context report
    print(visualizer.generate_context_report())

if __name__ == "__main__":
    main()