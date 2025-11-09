#!/usr/bin/env python3
"""
Unified Launcher for SOTA Neural Chatbot
Launch the complete continually-learning chatbot system
"""

import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🧠 SOTA Neural Network Chatbot with Continual Learning    ║
║                                                              ║
║    Features:                                                 ║
║    • Transformer-based neural architecture                   ║
║    • Retrieval Augmented Generation (RAG)                    ║
║    • Continual learning with experience replay               ║
║    • Knowledge graph with vector storage                     ║
║    • Real-time web interface                                 ║
║    • Automated training orchestration                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import transformers
        import flask
        import flask_socketio
        import sentence_transformers
        import schedule
        logger.info("✓ All core dependencies found")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False


def launch_chatbot_server(args):
    """Launch the web-based chatbot server"""
    from chatbot_server import run_chatbot_server

    print(BANNER)
    print("\n🚀 Launching Chatbot Server...\n")

    run_chatbot_server(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


def launch_training_mode(args):
    """Launch training orchestrator"""
    from rag_system import create_rag_system
    from training_orchestrator import create_training_orchestrator

    print(BANNER)
    print("\n🎓 Launching Training Mode...\n")

    # Create RAG system
    logger.info("Initializing RAG system...")
    rag = create_rag_system(db_path=args.db_path)

    # Create training orchestrator
    logger.info("Initializing training orchestrator...")
    orchestrator = create_training_orchestrator(rag)

    # Setup training schedule
    orchestrator.setup_training_schedule(
        quick_learning_minutes=args.quick_minutes,
        full_learning_hours=args.full_hours,
        checkpoint_hours=args.checkpoint_hours
    )

    # Load checkpoint if specified
    if args.load_checkpoint:
        orchestrator.load_checkpoint(args.load_checkpoint)

    # Run scheduler
    if args.run_once:
        logger.info("Running all scheduled jobs once...")
        orchestrator.run_scheduler(run_once=True)
    elif args.manual_steps:
        logger.info(f"Running manual training session ({args.manual_steps} steps)...")
        metrics = orchestrator.manual_training_session(
            num_steps=args.manual_steps,
            batch_size=args.batch_size
        )
        print("\nTraining Metrics:")
        for key, value in metrics.items():
            if key != 'timestamp':
                print(f"  {key}: {value}")
    else:
        logger.info("Starting continuous training scheduler...")
        logger.info("Press Ctrl+C to stop")
        orchestrator.run_scheduler()


def launch_api_mode(args):
    """Launch REST API server"""
    from api_server import run_server

    print(BANNER)
    print("\n🔌 Launching API Server...\n")

    run_server(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


def launch_cli_mode(args):
    """Launch command-line interface"""
    from rag_system import create_rag_system

    print(BANNER)
    print("\n💬 Launching CLI Mode...\n")

    # Create RAG system
    logger.info("Initializing RAG system...")
    rag = create_rag_system(db_path=args.db_path)
    logger.info("✓ System ready!\n")

    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'stats' to see system statistics")
    print("=" * 60 + "\n")

    conversation_history = []
    user_id = "cli_user"

    try:
        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Your conversations have been saved for learning.")
                break

            if user_input.lower() == 'stats':
                stats = rag.get_comprehensive_stats()
                print("\n" + "=" * 60)
                print("SYSTEM STATISTICS")
                print("=" * 60)
                print(f"Total Conversations: {stats['chatbot']['total_conversations']}")
                print(f"Knowledge Items: {stats['chatbot']['knowledge_items']}")
                print(f"Episodic Memories: {stats['chatbot']['episodic_memory_size']}")
                print(f"Experience Buffer: {stats['chatbot']['experience_buffer_size']}")
                print(f"Knowledge Nodes: {stats['knowledge_store']['total_knowledge_nodes']}")
                print("=" * 60 + "\n")
                continue

            if not user_input:
                continue

            # Generate response
            result = rag.generate_with_retrieval(
                user_input=user_input,
                conversation_history=conversation_history,
                user_id=user_id
            )

            # Update history
            conversation_history.append({'role': 'user', 'content': user_input})
            conversation_history.append({'role': 'bot', 'content': result['response']})

            # Display response
            print(f"\nBot: {result['response']}")
            print(f"[Confidence: {result['confidence']:.0%} | Sources: {result['retrieved_knowledge_count']}]\n")

            # Ask for feedback
            if args.feedback:
                feedback = input("Was this helpful? (y/n/skip): ").strip().lower()
                if feedback == 'y':
                    rag.chatbot.learn_from_interaction(
                        user_input, result['response'], rating=1.0
                    )
                    print("✓ Thank you! I learned from this interaction.\n")
                elif feedback == 'n':
                    rag.chatbot.learn_from_interaction(
                        user_input, result['response'], rating=0.0
                    )
                    print("✓ Thank you for the feedback. I'll try to improve.\n")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Your conversations have been saved for learning.")
    except Exception as e:
        logger.error(f"Error: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='SOTA Neural Network Chatbot Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web chatbot server
  python launch_chatbot.py server

  # Launch CLI chatbot
  python launch_chatbot.py cli

  # Launch training mode
  python launch_chatbot.py train --manual-steps 20

  # Launch API server
  python launch_chatbot.py api --port 8000

  # Run scheduled training
  python launch_chatbot.py train --quick-minutes 5 --full-hours 1
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Server mode
    server_parser = subparsers.add_parser('server', help='Launch web chatbot server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    server_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # CLI mode
    cli_parser = subparsers.add_parser('cli', help='Launch CLI chatbot')
    cli_parser.add_argument('--db-path', default='chatbot_knowledge.db', help='Database path')
    cli_parser.add_argument('--feedback', action='store_true', help='Ask for feedback after responses')

    # Training mode
    train_parser = subparsers.add_parser('train', help='Launch training mode')
    train_parser.add_argument('--db-path', default='chatbot_knowledge.db', help='Database path')
    train_parser.add_argument('--quick-minutes', type=int, default=5, help='Minutes between quick learning')
    train_parser.add_argument('--full-hours', type=int, default=1, help='Hours between full learning')
    train_parser.add_argument('--checkpoint-hours', type=int, default=6, help='Hours between checkpoints')
    train_parser.add_argument('--manual-steps', type=int, help='Run manual training for N steps')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--run-once', action='store_true', help='Run all scheduled jobs once and exit')
    train_parser.add_argument('--load-checkpoint', help='Load checkpoint file')

    # API mode
    api_parser = subparsers.add_parser('api', help='Launch REST API server')
    api_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    api_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Execute command
    if args.command == 'server':
        launch_chatbot_server(args)
    elif args.command == 'cli':
        launch_cli_mode(args)
    elif args.command == 'train':
        launch_training_mode(args)
    elif args.command == 'api':
        launch_api_mode(args)
    else:
        parser.print_help()
        print("\nNo command specified. Use 'server', 'cli', 'train', or 'api'")
        sys.exit(1)


if __name__ == '__main__':
    main()
