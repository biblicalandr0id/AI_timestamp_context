"""
State-of-the-Art Neural Network Chatbot with Continual Learning
Advanced transformer-based architecture with memory and knowledge storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import json
import pickle
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatbotConfig:
    """Configuration for the neural chatbot"""
    # Model architecture
    model_name: str = "microsoft/DialoGPT-small"  # Can upgrade to medium/large
    max_length: int = 512
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Continual learning
    experience_replay_size: int = 1000
    ewc_lambda: float = 0.4  # Elastic Weight Consolidation
    rehearsal_ratio: float = 0.3  # Ratio of old examples to replay

    # Memory systems
    episodic_memory_size: int = 500
    semantic_memory_size: int = 1000
    working_memory_size: int = 10

    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2

    # Knowledge management
    min_confidence_for_storage: float = 0.6
    knowledge_update_threshold: float = 0.7


class ExperienceReplayBuffer:
    """Experience replay buffer for continual learning"""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)

    def add(self, conversation: Dict[str, Any], importance: float = 1.0):
        """Add conversation to replay buffer"""
        self.buffer.append(conversation)
        self.priorities.append(importance)

    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Sample n conversations from buffer (prioritized)"""
        if len(self.buffer) == 0:
            return []

        # Convert to numpy for sampling
        priorities = np.array(self.priorities)
        priorities = priorities / priorities.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=min(n, len(self.buffer)),
            replace=False,
            p=priorities
        )

        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class EpisodicMemory:
    """Episodic memory for specific conversation instances"""

    def __init__(self, max_size: int = 500):
        self.memories = deque(maxlen=max_size)
        self.embeddings = None

    def add_episode(
        self,
        user_input: str,
        bot_response: str,
        context: Dict[str, Any],
        embedding: np.ndarray
    ):
        """Store an episode"""
        episode = {
            'user_input': user_input,
            'bot_response': bot_response,
            'context': context,
            'timestamp': datetime.utcnow(),
            'embedding': embedding
        }
        self.memories.append(episode)

    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve similar episodes"""
        if len(self.memories) == 0:
            return []

        embeddings = np.array([m['embedding'] for m in self.memories])

        # Cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [self.memories[i] for i in top_indices]


class SemanticMemory:
    """Semantic memory for general knowledge"""

    def __init__(self, max_size: int = 1000):
        self.knowledge_base = {}
        self.max_size = max_size

    def add_knowledge(self, key: str, value: Any, confidence: float = 1.0):
        """Add knowledge to semantic memory"""
        if key in self.knowledge_base:
            # Update existing knowledge with weighted average
            old_conf = self.knowledge_base[key]['confidence']
            old_val = self.knowledge_base[key]['value']

            new_conf = (old_conf + confidence) / 2
            # If values differ, keep track of both
            self.knowledge_base[key] = {
                'value': value,
                'previous_value': old_val,
                'confidence': new_conf,
                'updated': datetime.utcnow()
            }
        else:
            # Enforce size limit
            if len(self.knowledge_base) >= self.max_size:
                # Remove lowest confidence entry
                min_key = min(
                    self.knowledge_base.keys(),
                    key=lambda k: self.knowledge_base[k]['confidence']
                )
                del self.knowledge_base[min_key]

            self.knowledge_base[key] = {
                'value': value,
                'confidence': confidence,
                'created': datetime.utcnow(),
                'updated': datetime.utcnow()
            }

    def retrieve(self, key: str) -> Optional[Dict]:
        """Retrieve knowledge"""
        return self.knowledge_base.get(key)

    def search(self, query: str, threshold: float = 0.5) -> List[Tuple[str, Dict]]:
        """Search knowledge base"""
        results = []
        query_lower = query.lower()

        for key, value in self.knowledge_base.items():
            if query_lower in key.lower():
                if value['confidence'] >= threshold:
                    results.append((key, value))

        return sorted(results, key=lambda x: x[1]['confidence'], reverse=True)


class NeuralChatbot(nn.Module):
    """
    State-of-the-art neural chatbot with continual learning
    """

    def __init__(self, config: ChatbotConfig):
        super().__init__()
        self.config = config

        # Load pre-trained model
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)

        # Add special tokens if needed
        special_tokens = {'pad_token': '<pad>', 'sep_token': '<sep>'}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Encoder for embeddings (separate from generation)
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Memory systems
        self.episodic_memory = EpisodicMemory(config.episodic_memory_size)
        self.semantic_memory = SemanticMemory(config.semantic_memory_size)
        self.working_memory = deque(maxlen=config.working_memory_size)

        # Experience replay for continual learning
        self.experience_buffer = ExperienceReplayBuffer(config.experience_replay_size)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        # For EWC (Elastic Weight Consolidation)
        self.fisher_information = None
        self.optimal_params = None

        # Statistics
        self.stats = {
            'total_conversations': 0,
            'total_learned_interactions': 0,
            'knowledge_items': 0,
            'average_confidence': 0.0
        }

        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.encoder.to(self.device)

        logger.info(f"Chatbot initialized on device: {self.device}")

    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        inputs = self.encoder_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0]

    def generate_response(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]] = None,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Generate response to user input

        Returns dict with 'response', 'confidence', 'memory_used', etc.
        """
        # Add to working memory
        self.working_memory.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.utcnow()
        })

        # Get embedding for memory retrieval
        query_embedding = self.encode_text(user_input)

        # Retrieve relevant memories
        relevant_episodes = []
        if use_memory and len(self.episodic_memory.memories) > 0:
            relevant_episodes = self.episodic_memory.retrieve_similar(
                query_embedding, top_k=3
            )

        # Build context with conversation history
        context = self._build_context(
            user_input,
            conversation_history or [],
            relevant_episodes
        )

        # Generate response
        response_text, confidence = self._generate(context)

        # Add to working memory
        self.working_memory.append({
            'role': 'bot',
            'content': response_text,
            'timestamp': datetime.utcnow()
        })

        # Store in episodic memory
        self.episodic_memory.add_episode(
            user_input=user_input,
            bot_response=response_text,
            context={'conversation_history': conversation_history},
            embedding=query_embedding
        )

        # Extract and store knowledge
        if confidence >= self.config.min_confidence_for_storage:
            self._extract_and_store_knowledge(user_input, response_text, confidence)

        # Update statistics
        self.stats['total_conversations'] += 1

        return {
            'response': response_text,
            'confidence': confidence,
            'memory_used': len(relevant_episodes) > 0,
            'relevant_memories': len(relevant_episodes),
            'working_memory_size': len(self.working_memory)
        }

    def _build_context(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        relevant_episodes: List[Dict]
    ) -> str:
        """Build context for generation"""
        context_parts = []

        # Add relevant memories
        if relevant_episodes:
            context_parts.append("Relevant past conversations:")
            for ep in relevant_episodes[:2]:  # Limit to avoid context overflow
                context_parts.append(f"User: {ep['user_input']}")
                context_parts.append(f"Bot: {ep['bot_response']}")
            context_parts.append("")

        # Add recent conversation history
        for turn in conversation_history[-5:]:  # Last 5 turns
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            context_parts.append(f"{role.capitalize()}: {content}")

        # Add current input
        context_parts.append(f"User: {user_input}")
        context_parts.append("Bot:")

        return "\n".join(context_parts)

    def _generate(self, context: str) -> Tuple[str, float]:
        """Generate response using the model"""
        # Tokenize
        inputs = self.tokenizer(
            context,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode
        response = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Calculate confidence from generation scores
        if hasattr(outputs, 'scores') and outputs.scores:
            scores = torch.stack(outputs.scores, dim=1)
            probs = F.softmax(scores, dim=-1)
            max_probs = probs.max(dim=-1).values
            confidence = max_probs.mean().item()
        else:
            confidence = 0.5  # Default

        return response.strip(), confidence

    def _extract_and_store_knowledge(
        self,
        user_input: str,
        bot_response: str,
        confidence: float
    ):
        """Extract knowledge from interaction and store"""
        # Simple knowledge extraction (can be enhanced)
        input_lower = user_input.lower()

        # Pattern: "what is X?" -> store knowledge about X
        if 'what is' in input_lower or 'what are' in input_lower:
            # Extract topic
            topic = user_input.split('what is')[-1].split('what are')[-1].strip('? ')
            if topic:
                self.semantic_memory.add_knowledge(
                    key=f"definition:{topic}",
                    value=bot_response,
                    confidence=confidence
                )
                self.stats['knowledge_items'] += 1

        # Pattern: "how to X?" -> store procedural knowledge
        elif 'how to' in input_lower or 'how do' in input_lower:
            topic = user_input.split('how to')[-1].split('how do')[-1].strip('? ')
            if topic:
                self.semantic_memory.add_knowledge(
                    key=f"procedure:{topic}",
                    value=bot_response,
                    confidence=confidence
                )
                self.stats['knowledge_items'] += 1

    def learn_from_interaction(
        self,
        user_input: str,
        bot_response: str,
        user_feedback: Optional[str] = None,
        rating: Optional[float] = None
    ):
        """
        Learn from a completed interaction

        Args:
            user_input: What the user said
            bot_response: What the bot responded
            user_feedback: Optional feedback from user
            rating: Optional rating (0-1)
        """
        # Calculate importance for replay buffer
        importance = rating if rating is not None else 0.5

        # Adjust importance based on feedback
        if user_feedback:
            feedback_lower = user_feedback.lower()
            if any(word in feedback_lower for word in ['good', 'great', 'perfect', 'thanks']):
                importance = min(1.0, importance + 0.3)
            elif any(word in feedback_lower for word in ['bad', 'wrong', 'incorrect']):
                importance = max(0.1, importance - 0.3)

        # Add to experience replay buffer
        conversation = {
            'user_input': user_input,
            'bot_response': bot_response,
            'feedback': user_feedback,
            'rating': rating,
            'timestamp': datetime.utcnow()
        }
        self.experience_buffer.add(conversation, importance)

        self.stats['total_learned_interactions'] += 1

        logger.info(f"Learned from interaction (importance: {importance:.2f})")

    def continual_learning_step(self, batch_size: int = None):
        """
        Perform one step of continual learning using experience replay
        """
        if len(self.experience_buffer) < 4:
            logger.info("Not enough experiences for continual learning")
            return

        batch_size = batch_size or self.config.batch_size

        # Sample from experience buffer
        experiences = self.experience_buffer.sample(batch_size)

        # Prepare training data
        contexts = []
        targets = []

        for exp in experiences:
            context = f"User: {exp['user_input']}\nBot:"
            target = exp['bot_response']
            contexts.append(context)
            targets.append(target)

        # Tokenize
        context_encodings = self.tokenizer(
            contexts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)

        target_encodings = self.tokenizer(
            targets,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)

        # Combine input and target
        input_ids = torch.cat([
            context_encodings['input_ids'],
            target_encodings['input_ids']
        ], dim=1)

        attention_mask = torch.cat([
            context_encodings['attention_mask'],
            target_encodings['attention_mask']
        ], dim=1)

        # Create labels (shift right)
        labels = input_ids.clone()
        labels[:, :context_encodings['input_ids'].shape[1]] = -100  # Don't compute loss on context

        # Forward pass
        self.model.train()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Add EWC regularization if available
        if self.fisher_information is not None and self.optimal_params is not None:
            ewc_loss = self._compute_ewc_loss()
            loss = loss + self.config.ewc_lambda * ewc_loss

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Update
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.model.eval()

        logger.info(f"Continual learning step completed. Loss: {loss.item():.4f}")

        return loss.item()

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Compute Elastic Weight Consolidation loss"""
        ewc_loss = 0.0

        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()

        return ewc_loss

    def compute_fisher_information(self, num_samples: int = 100):
        """Compute Fisher Information Matrix for EWC"""
        logger.info("Computing Fisher Information Matrix...")

        self.fisher_information = {}
        self.optimal_params = {}

        # Store current parameters
        for name, param in self.model.named_parameters():
            self.optimal_params[name] = param.data.clone()
            self.fisher_information[name] = torch.zeros_like(param.data)

        # Sample experiences
        if len(self.experience_buffer) == 0:
            return

        samples = self.experience_buffer.sample(min(num_samples, len(self.experience_buffer)))

        self.model.train()

        for exp in samples:
            context = f"User: {exp['user_input']}\nBot:"
            target = exp['bot_response']

            # Tokenize
            inputs = self.tokenizer(
                context + target,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)

            # Forward
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Accumulate gradients squared (Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2 / len(samples)

        self.model.eval()
        logger.info("Fisher Information Matrix computed")

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stats': self.stats,
            'fisher_information': self.fisher_information,
            'optimal_params': self.optimal_params
        }

        torch.save(checkpoint, filepath)

        # Save memories separately (they can be large)
        memory_filepath = filepath.replace('.pt', '_memories.pkl')
        memories = {
            'episodic': self.episodic_memory,
            'semantic': self.semantic_memory,
            'experience_buffer': self.experience_buffer
        }

        with open(memory_filepath, 'wb') as f:
            pickle.dump(memories, f)

        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats']
        self.fisher_information = checkpoint.get('fisher_information')
        self.optimal_params = checkpoint.get('optimal_params')

        # Load memories
        memory_filepath = filepath.replace('.pt', '_memories.pkl')
        try:
            with open(memory_filepath, 'rb') as f:
                memories = pickle.load(f)
                self.episodic_memory = memories['episodic']
                self.semantic_memory = memories['semantic']
                self.experience_buffer = memories['experience_buffer']
        except FileNotFoundError:
            logger.warning("Memory file not found, starting with empty memories")

        logger.info(f"Checkpoint loaded from {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        return {
            **self.stats,
            'episodic_memory_size': len(self.episodic_memory.memories),
            'semantic_memory_size': len(self.semantic_memory.knowledge_base),
            'experience_buffer_size': len(self.experience_buffer),
            'working_memory_size': len(self.working_memory),
            'device': str(self.device)
        }


def create_chatbot(config: Optional[ChatbotConfig] = None) -> NeuralChatbot:
    """Create a new chatbot instance"""
    if config is None:
        config = ChatbotConfig()

    return NeuralChatbot(config)


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("SOTA NEURAL CHATBOT WITH CONTINUAL LEARNING")
    print("=" * 60)

    # Create chatbot
    config = ChatbotConfig()
    chatbot = create_chatbot(config)

    print(f"\nChatbot created successfully!")
    print(f"Device: {chatbot.device}")
    print(f"Model: {config.model_name}")

    # Example conversation
    print("\n" + "-" * 60)
    print("EXAMPLE CONVERSATION")
    print("-" * 60)

    conversation_history = []

    # Turn 1
    user_input = "Hello! What is artificial intelligence?"
    print(f"\nUser: {user_input}")

    result = chatbot.generate_response(user_input, conversation_history)
    print(f"Bot: {result['response']}")
    print(f"Confidence: {result['confidence']:.2f}")

    conversation_history.append({'role': 'user', 'content': user_input})
    conversation_history.append({'role': 'bot', 'content': result['response']})

    # Learn from interaction
    chatbot.learn_from_interaction(user_input, result['response'], rating=0.8)

    # Perform continual learning
    chatbot.continual_learning_step()

    # Show stats
    print("\n" + "=" * 60)
    print("CHATBOT STATISTICS")
    print("=" * 60)
    stats = chatbot.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
