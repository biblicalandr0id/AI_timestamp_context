"""
Training Orchestrator for Continual Learning
Manages training schedules, checkpointing, and model optimization
"""

import torch
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
from pathlib import Path
import json

from neural_chatbot import NeuralChatbot, ChatbotConfig
from knowledge_store import KnowledgeGraphManager
from rag_system import RetrievalAugmentedGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates continual learning with scheduling and optimization
    """

    def __init__(
        self,
        rag_system: RetrievalAugmentedGenerator,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.rag_system = rag_system
        self.chatbot = rag_system.chatbot
        self.knowledge_manager = rag_system.knowledge_manager

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.training_history = []
        self.last_checkpoint = None

        # Training schedule
        self.schedule_active = False

        logger.info("Training Orchestrator initialized")

    def setup_training_schedule(
        self,
        quick_learning_minutes: int = 5,
        full_learning_hours: int = 1,
        checkpoint_hours: int = 6,
        fisher_days: int = 1
    ):
        """
        Setup automated training schedule

        Args:
            quick_learning_minutes: Minutes between quick learning cycles
            full_learning_hours: Hours between full learning sessions
            checkpoint_hours: Hours between checkpoints
            fisher_days: Days between Fisher Information Matrix updates
        """
        # Quick learning every N minutes
        schedule.every(quick_learning_minutes).minutes.do(
            self.quick_learning_cycle
        )

        # Full learning session every N hours
        schedule.every(full_learning_hours).hours.do(
            self.full_learning_session
        )

        # Checkpoint every N hours
        schedule.every(checkpoint_hours).hours.do(
            self.create_checkpoint
        )

        # Update Fisher Information for EWC
        schedule.every(fisher_days).days.do(
            self.update_fisher_information
        )

        self.schedule_active = True

        logger.info(f"Training schedule setup:")
        logger.info(f"  - Quick learning: every {quick_learning_minutes} minutes")
        logger.info(f"  - Full learning: every {full_learning_hours} hours")
        logger.info(f"  - Checkpoints: every {checkpoint_hours} hours")
        logger.info(f"  - Fisher update: every {fisher_days} days")

    def quick_learning_cycle(self):
        """Quick learning cycle from recent interactions"""
        try:
            logger.info("Starting quick learning cycle...")

            # Learn from recent conversations (batch size 5)
            self.rag_system.batch_learn_from_conversations(batch_size=5)

            # Perform 2-3 continual learning steps
            losses = []
            for _ in range(2):
                loss = self.chatbot.continual_learning_step(batch_size=4)
                if loss:
                    losses.append(loss)

            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # Record in history
            self.training_history.append({
                'type': 'quick',
                'timestamp': datetime.utcnow(),
                'avg_loss': avg_loss,
                'steps': len(losses)
            })

            logger.info(f"Quick learning cycle completed. Avg loss: {avg_loss:.4f}")

        except Exception as e:
            logger.error(f"Error in quick learning cycle: {e}")

    def full_learning_session(self):
        """Full learning session with more iterations"""
        try:
            logger.info("Starting full learning session...")

            # Learn from all unlearned conversations
            self.rag_system.batch_learn_from_conversations(batch_size=20)

            # Perform more continual learning steps
            losses = []
            for _ in range(10):
                loss = self.chatbot.continual_learning_step(batch_size=8)
                if loss:
                    losses.append(loss)

            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # Record in history
            self.training_history.append({
                'type': 'full',
                'timestamp': datetime.utcnow(),
                'avg_loss': avg_loss,
                'steps': len(losses)
            })

            logger.info(f"Full learning session completed. Avg loss: {avg_loss:.4f}")

            # Update statistics
            self.chatbot.stats['average_confidence'] = avg_loss

        except Exception as e:
            logger.error(f"Error in full learning session: {e}")

    def update_fisher_information(self):
        """Update Fisher Information Matrix for Elastic Weight Consolidation"""
        try:
            logger.info("Updating Fisher Information Matrix...")

            self.chatbot.compute_fisher_information(num_samples=100)

            logger.info("Fisher Information Matrix updated")

        except Exception as e:
            logger.error(f"Error updating Fisher Information: {e}")

    def create_checkpoint(self, name: Optional[str] = None):
        """Create a model checkpoint"""
        try:
            if name is None:
                name = f"checkpoint_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pt"

            filepath = self.checkpoint_dir / name

            logger.info(f"Creating checkpoint: {filepath}")

            # Save chatbot checkpoint
            self.chatbot.save_checkpoint(str(filepath))

            self.last_checkpoint = {
                'filepath': str(filepath),
                'timestamp': datetime.utcnow(),
                'stats': self.chatbot.get_stats()
            }

            # Save training history
            history_file = filepath.with_suffix('.json')
            with open(history_file, 'w') as f:
                json.dump({
                    'training_history': [
                        {**h, 'timestamp': h['timestamp'].isoformat()}
                        for h in self.training_history
                    ],
                    'checkpoint_info': {
                        **self.last_checkpoint,
                        'timestamp': self.last_checkpoint['timestamp'].isoformat()
                    }
                }, f, indent=2)

            logger.info(f"Checkpoint created successfully")

            # Keep only last 10 checkpoints
            self._cleanup_old_checkpoints(keep=10)

        except Exception as e:
            logger.error(f"Error creating checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """Load a model checkpoint"""
        try:
            logger.info(f"Loading checkpoint: {filepath}")

            self.chatbot.load_checkpoint(filepath)

            # Load training history if available
            history_file = Path(filepath).with_suffix('.json')
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.training_history = [
                        {**h, 'timestamp': datetime.fromisoformat(h['timestamp'])}
                        for h in data.get('training_history', [])
                    ]

            logger.info("Checkpoint loaded successfully")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    def _cleanup_old_checkpoints(self, keep: int = 10):
        """Keep only the N most recent checkpoints"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for checkpoint in checkpoints[keep:]:
            try:
                checkpoint.unlink()
                # Also remove associated files
                checkpoint.with_suffix('.json').unlink(missing_ok=True)
                checkpoint_str = str(checkpoint).replace('.pt', '_memories.pkl')
                Path(checkpoint_str).unlink(missing_ok=True)

                logger.info(f"Removed old checkpoint: {checkpoint.name}")
            except Exception as e:
                logger.error(f"Error removing checkpoint {checkpoint}: {e}")

    def run_scheduler(self, run_once: bool = False):
        """
        Run the training scheduler

        Args:
            run_once: If True, run all pending jobs once and return
        """
        if not self.schedule_active:
            logger.warning("Training schedule not setup. Call setup_training_schedule() first.")
            return

        logger.info("Training scheduler started")

        try:
            if run_once:
                schedule.run_all()
                logger.info("All scheduled jobs executed once")
            else:
                while True:
                    schedule.run_pending()
                    time.sleep(30)  # Check every 30 seconds

        except KeyboardInterrupt:
            logger.info("Training scheduler stopped by user")
        except Exception as e:
            logger.error(f"Error in training scheduler: {e}")

    def manual_training_session(
        self,
        num_steps: int = 10,
        batch_size: int = 8,
        create_checkpoint_after: bool = True
    ) -> Dict:
        """
        Manually run a training session

        Args:
            num_steps: Number of training steps
            batch_size: Batch size for training
            create_checkpoint_after: Whether to create a checkpoint after training

        Returns:
            Training metrics
        """
        logger.info(f"Starting manual training session ({num_steps} steps)...")

        losses = []
        start_time = time.time()

        for step in range(num_steps):
            loss = self.chatbot.continual_learning_step(batch_size=batch_size)
            if loss:
                losses.append(loss)
                logger.info(f"  Step {step+1}/{num_steps}: loss={loss:.4f}")

        duration = time.time() - start_time

        metrics = {
            'num_steps': num_steps,
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
            'min_loss': min(losses) if losses else 0.0,
            'max_loss': max(losses) if losses else 0.0,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow()
        }

        self.training_history.append({
            'type': 'manual',
            **metrics,
            'timestamp': metrics['timestamp']
        })

        if create_checkpoint_after:
            self.create_checkpoint()

        logger.info(f"Manual training session completed. Avg loss: {metrics['avg_loss']:.4f}")

        return metrics

    def get_training_report(self) -> Dict:
        """Generate training report"""
        if not self.training_history:
            return {'message': 'No training history available'}

        # Calculate statistics
        recent_history = self.training_history[-100:]  # Last 100 sessions

        total_sessions = len(self.training_history)
        quick_sessions = len([h for h in self.training_history if h['type'] == 'quick'])
        full_sessions = len([h for h in self.training_history if h['type'] == 'full'])
        manual_sessions = len([h for h in self.training_history if h['type'] == 'manual'])

        avg_loss_recent = sum(h['avg_loss'] for h in recent_history) / len(recent_history)
        avg_loss_all = sum(h['avg_loss'] for h in self.training_history) / total_sessions

        # Learning trend
        if len(recent_history) >= 20:
            first_10_avg = sum(h['avg_loss'] for h in recent_history[:10]) / 10
            last_10_avg = sum(h['avg_loss'] for h in recent_history[-10:]) / 10
            improvement = ((first_10_avg - last_10_avg) / first_10_avg * 100) if first_10_avg > 0 else 0
        else:
            improvement = 0.0

        return {
            'total_training_sessions': total_sessions,
            'session_breakdown': {
                'quick': quick_sessions,
                'full': full_sessions,
                'manual': manual_sessions
            },
            'average_loss': {
                'recent_100': avg_loss_recent,
                'all_time': avg_loss_all
            },
            'improvement_percentage': improvement,
            'last_checkpoint': self.last_checkpoint,
            'chatbot_stats': self.chatbot.get_stats(),
            'knowledge_stats': self.knowledge_manager.get_statistics()
        }

    def optimize_model(self):
        """Optimize model for inference (quantization, pruning, etc.)"""
        logger.info("Optimizing model for inference...")

        try:
            # Quantize model for faster inference
            if hasattr(torch.quantization, 'quantize_dynamic'):
                self.chatbot.model = torch.quantization.quantize_dynamic(
                    self.chatbot.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Model quantized successfully")

        except Exception as e:
            logger.error(f"Error optimizing model: {e}")


def create_training_orchestrator(rag_system: RetrievalAugmentedGenerator) -> TrainingOrchestrator:
    """Create a training orchestrator"""
    return TrainingOrchestrator(rag_system)


if __name__ == '__main__':
    print("=" * 60)
    print("TRAINING ORCHESTRATOR TEST")
    print("=" * 60)

    from rag_system import create_rag_system

    # Create RAG system
    rag = create_rag_system()

    # Create orchestrator
    orchestrator = create_training_orchestrator(rag)

    # Setup training schedule
    orchestrator.setup_training_schedule(
        quick_learning_minutes=5,
        full_learning_hours=1,
        checkpoint_hours=6
    )

    # Run manual training session as test
    print("\n" + "-" * 60)
    print("RUNNING MANUAL TRAINING SESSION")
    print("-" * 60)

    metrics = orchestrator.manual_training_session(num_steps=5, batch_size=4)

    print(f"\nTraining Metrics:")
    for key, value in metrics.items():
        if key != 'timestamp':
            print(f"  {key}: {value}")

    # Get training report
    print("\n" + "=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)

    report = orchestrator.get_training_report()
    print(json.dumps(report, indent=2, default=str))

    print("\n✓ Training orchestrator test complete!")
