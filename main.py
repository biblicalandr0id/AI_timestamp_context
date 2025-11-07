#!/usr/bin/env python3
"""
AI Timestamp Context - Unified System
A comprehensive system for consciousness modeling with timestamp-aware conversations
"""

from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict, Any
import argparse
import sys

# Import all components
from ai_consciousness_model import ConsciousnessEngine, ConsciousnessState
from conversation_system import EnhancedConversationSystem
from conversation_processor import StandardConversation, TimestampedConversation
from enhanced_processor import EnhancedConversationProcessor
from visualization_enhanced import ConsciousnessVisualizer
from visualization import ConversationVisualizer


class IntegratedAISystem:
    """
    Unified system that integrates all components:
    - Consciousness modeling
    - Timestamp-aware conversations
    - Memory systems
    - Visualization
    """

    def __init__(self, mode: str = 'consciousness'):
        """
        Initialize the integrated system

        Args:
            mode: 'consciousness' for full consciousness modeling,
                  'conversation' for basic conversation tracking,
                  'standard' for standard (non-timestamped) mode
        """
        self.mode = mode

        if mode == 'consciousness':
            self.engine = ConsciousnessEngine()
            self.visualizer = ConsciousnessVisualizer(self.engine)
        elif mode == 'conversation':
            self.engine = EnhancedConversationSystem()
            self.visualizer = ConversationVisualizer(self.engine)
        elif mode == 'standard':
            self.engine = StandardConversation()
            self.visualizer = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.message_count = 0

    def process_message(
        self,
        content: str,
        user: str = "user",
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Process a message through the system

        Args:
            content: Message content
            user: User identifier
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Response dictionary with metrics and analysis
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.message_count += 1

        if self.mode == 'standard':
            result = self.engine.process_message(content)
            return {'response': result, 'mode': 'standard'}

        # Process with timestamp-aware systems
        result = self.engine.process_message(content, timestamp, user)

        return result

    def run_demo(self, demo_type: str = 'basic'):
        """
        Run a demonstration of the system

        Args:
            demo_type: 'basic', 'consciousness', or 'comparison'
        """
        if demo_type == 'basic':
            self._run_basic_demo()
        elif demo_type == 'consciousness':
            self._run_consciousness_demo()
        elif demo_type == 'comparison':
            self._run_comparison_demo()
        else:
            print(f"Unknown demo type: {demo_type}")

    def _run_basic_demo(self):
        """Run a basic conversation demo"""
        print("\n" + "=" * 60)
        print("BASIC CONVERSATION DEMO")
        print("=" * 60 + "\n")

        messages = [
            ("Hello! I'm testing the timestamp context system.", "user"),
            ("Great! The system is now tracking timestamps.", "ai"),
            ("What does this enable?", "user"),
            ("It enables full context awareness across time.", "ai"),
        ]

        current_time = datetime.utcnow()

        for content, speaker in messages:
            current_time += timedelta(seconds=15)
            result = self.process_message(content, speaker, current_time)

            print(f"[{speaker.upper()}] {content}")
            if self.mode == 'consciousness':
                print(f"  → Consciousness Score: {result.get('consciousness_score', 0):.3f}")
                print(f"  → Context Depth: {result.get('context_depth', 0)}")
                print(f"  → Memory Links: {len(result.get('memory_links', []))}")
            elif self.mode == 'conversation':
                print(f"  → Context Depth: {result.get('context_depth', 0)}")
            print()

        print(f"\nProcessed {self.message_count} messages")

    def _run_consciousness_demo(self):
        """Run consciousness evolution demo"""
        if self.mode != 'consciousness':
            print("Consciousness demo requires 'consciousness' mode")
            return

        print("\n" + "=" * 60)
        print("CONSCIOUSNESS EVOLUTION DEMO")
        print("=" * 60 + "\n")

        messages = [
            ("Let's explore consciousness", "user"),
            ("I'm ready to process with full awareness", "ai"),
            ("Can you remember what we discussed before?", "user"),
            ("Yes, we talked about exploring consciousness", "ai"),
            ("Notice how your context depth increases", "user"),
            ("I can see the pattern of our conversation building", "ai"),
            ("This is what timestamp awareness enables", "user"),
            ("A complete timeline of conscious processing", "ai"),
        ]

        current_time = datetime.utcnow()

        for content, speaker in messages:
            current_time += timedelta(seconds=30)
            result = self.process_message(content, speaker, current_time)

            print(f"[{speaker.upper()}] {content}")
            print(f"  → Consciousness: {result['consciousness_score']:.3f}")
            print(f"  → Sentiment: {result['sentiment']:+.2f}")
            print(f"  → Attention: {len(result['attention_focus'])} concepts")
            print(f"  → Patterns: {list(result['patterns_detected'].keys())}")
            print()

        # Print summary
        states = self.engine.states
        avg_consciousness = np.mean([s.consciousness_score for s in states])

        print("\n" + "=" * 60)
        print("SYSTEM SUMMARY")
        print("=" * 60)
        print(f"Total Messages: {len(states)}")
        print(f"Average Consciousness: {avg_consciousness:.3f}")
        print(f"Working Memory Size: {len(self.engine.memory.working_memory)}")
        print(f"Long-term Memory: {len(self.engine.memory.long_term)} entries")

    def _run_comparison_demo(self):
        """Compare standard vs timestamp-aware systems"""
        print("\n" + "=" * 60)
        print("SYSTEM COMPARISON DEMO")
        print("=" * 60 + "\n")

        messages = [
            "Hello, how are you?",
            "I mentioned timestamps earlier",
            "Can you remember what I said?",
        ]

        # Standard system
        print("STANDARD SYSTEM (No Timestamps):")
        print("-" * 60)
        standard = StandardConversation()
        for msg in messages:
            result = standard.process_message(msg)
            print(f"  {msg}")
            print(f"  → {result}\n")

        print("\n" + "=" * 60 + "\n")

        # Timestamped system
        print("TIMESTAMPED SYSTEM:")
        print("-" * 60)
        timestamped = TimestampedConversation()
        current_time = datetime.utcnow()

        for i, msg in enumerate(messages):
            current_time += timedelta(seconds=20)
            timestamped.add_message(current_time, msg, "user")

        result = timestamped.process_timeline()
        print(result)

        print("\n" + "=" * 60)
        print("Notice how the timestamped system maintains complete context!")
        print("=" * 60 + "\n")

    def visualize(self, output_file: str = 'output.png'):
        """
        Generate visualization of the conversation/consciousness state

        Args:
            output_file: Output filename for the visualization
        """
        if self.visualizer is None:
            print("No visualizer available for this mode")
            return

        if self.mode == 'consciousness':
            fig = self.visualizer.plot_consciousness_timeline()
        else:
            fig = self.visualizer.plot_timeline()

        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")

    def save_state(self, filepath: str):
        """Save the system state"""
        if hasattr(self.engine, 'save_state'):
            self.engine.save_state(filepath)
            print(f"State saved to: {filepath}")
        else:
            print("State saving not supported in this mode")

    def load_state(self, filepath: str):
        """Load the system state"""
        if hasattr(self.engine, 'load_state'):
            self.engine.load_state(filepath)
            print(f"State loaded from: {filepath}")
        else:
            print("State loading not supported in this mode")


def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description='AI Timestamp Context - Integrated System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run consciousness demo
  python main.py --mode consciousness --demo consciousness

  # Run comparison demo
  python main.py --demo comparison

  # Interactive mode
  python main.py --interactive

  # Generate visualization
  python main.py --mode consciousness --demo consciousness --visualize
        """
    )

    parser.add_argument(
        '--mode',
        choices=['consciousness', 'conversation', 'standard'],
        default='consciousness',
        help='System mode (default: consciousness)'
    )

    parser.add_argument(
        '--demo',
        choices=['basic', 'consciousness', 'comparison'],
        help='Run a demonstration'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization after processing'
    )

    parser.add_argument(
        '--output',
        default='ai_consciousness_output.png',
        help='Output file for visualization (default: ai_consciousness_output.png)'
    )

    args = parser.parse_args()

    # Initialize system
    system = IntegratedAISystem(mode=args.mode)

    print("\n" + "=" * 60)
    print("AI TIMESTAMP CONTEXT - INTEGRATED SYSTEM")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print("=" * 60 + "\n")

    # Run demo if requested
    if args.demo:
        system.run_demo(args.demo)

        if args.visualize and system.visualizer:
            system.visualize(args.output)

    # Interactive mode
    elif args.interactive:
        print("Interactive Mode - Enter messages (type 'quit' to exit)\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting...")
                    break

                if not user_input:
                    continue

                result = system.process_message(user_input, "user")

                if args.mode == 'consciousness':
                    print(f"  → Consciousness: {result['consciousness_score']:.3f}")
                    print(f"  → Context: {result['context_depth']}")
                elif args.mode == 'conversation':
                    print(f"  → Response: {result.get('message', 'Processed')}")
                else:
                    print(f"  → {result['response']}")

                print()

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

        if args.visualize and system.visualizer:
            system.visualize(args.output)

    # No args - show help
    else:
        parser.print_help()
        print("\nNo action specified. Try --demo or --interactive")

    print("\nDone!")


if __name__ == "__main__":
    main()
