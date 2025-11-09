"""
Retrieval Augmented Generation (RAG) System
Combines neural generation with knowledge retrieval for enhanced responses
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from neural_chatbot import NeuralChatbot, ChatbotConfig
from knowledge_store import KnowledgeGraphManager, KnowledgeNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    # Retrieval parameters
    retrieval_top_k: int = 5
    retrieval_min_confidence: float = 0.5
    max_context_tokens: int = 1024

    # Fusion parameters
    knowledge_weight: float = 0.7  # Weight for retrieved knowledge vs generated
    diversity_penalty: float = 0.1  # Penalty for redundant information
    recency_bias: float = 0.2  # Bias towards more recent knowledge

    # Quality thresholds
    min_relevance_score: float = 0.3
    min_combined_confidence: float = 0.6


class RetrievalAugmentedGenerator:
    """
    RAG system that enhances neural generation with knowledge retrieval
    """

    def __init__(
        self,
        chatbot: NeuralChatbot,
        knowledge_manager: KnowledgeGraphManager,
        config: Optional[RAGConfig] = None
    ):
        self.chatbot = chatbot
        self.knowledge_manager = knowledge_manager
        self.config = config or RAGConfig()

        self.device = chatbot.device

        logger.info("RAG system initialized")

    def generate_with_retrieval(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using RAG

        Args:
            user_input: User's message
            conversation_history: Previous conversation
            user_id: Optional user identifier

        Returns:
            Dict with response, sources, confidence, etc.
        """
        # Step 1: Generate query embedding
        query_embedding = self.chatbot.encode_text(user_input)

        # Step 2: Retrieve relevant knowledge
        retrieved_knowledge = self._retrieve_knowledge(
            query_embedding,
            user_input,
            user_id
        )

        # Step 3: Build augmented context
        augmented_context = self._build_augmented_context(
            user_input,
            conversation_history or [],
            retrieved_knowledge
        )

        # Step 4: Generate response with augmented context
        response_data = self._generate_augmented_response(
            user_input,
            augmented_context,
            retrieved_knowledge
        )

        # Step 5: Post-process and add metadata
        final_response = self._postprocess_response(
            response_data,
            retrieved_knowledge,
            query_embedding
        )

        # Step 6: Store interaction for future retrieval
        self._store_interaction(
            user_input,
            final_response['response'],
            query_embedding,
            final_response['confidence'],
            user_id
        )

        return final_response

    def _retrieve_knowledge(
        self,
        query_embedding: np.ndarray,
        user_input: str,
        user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the knowledge store"""
        # Retrieve from knowledge graph
        knowledge_nodes = self.knowledge_manager.retrieve_relevant_knowledge(
            query_embedding,
            top_k=self.config.retrieval_top_k,
            min_confidence=self.config.retrieval_min_confidence
        )

        # Retrieve from episodic memory
        episodic_memories = self.chatbot.episodic_memory.retrieve_similar(
            query_embedding,
            top_k=3
        )

        # Search semantic memory
        semantic_results = self.chatbot.semantic_memory.search(
            user_input,
            threshold=self.config.retrieval_min_confidence
        )

        # Get user profile if available
        user_profile = None
        if user_id:
            user_profile = self.knowledge_manager.store.get_user_profile(user_id)

        # Combine and score all retrieved knowledge
        retrieved = []

        # Add knowledge graph results
        for node in knowledge_nodes:
            retrieved.append({
                'content': node.content,
                'source': 'knowledge_graph',
                'confidence': node.confidence,
                'metadata': node.metadata,
                'node_id': node.id,
                'access_count': node.access_count
            })

        # Add episodic memories
        for memory in episodic_memories:
            retrieved.append({
                'content': f"Previous conversation:\nUser: {memory['user_input']}\nBot: {memory['bot_response']}",
                'source': 'episodic_memory',
                'confidence': 0.7,
                'metadata': memory.get('context', {}),
                'timestamp': memory.get('timestamp')
            })

        # Add semantic memory results
        for key, value in semantic_results[:3]:
            retrieved.append({
                'content': f"{key}: {value['value']}",
                'source': 'semantic_memory',
                'confidence': value['confidence'],
                'metadata': {'updated': value.get('updated')}
            })

        # Sort by confidence and recency
        retrieved = self._rerank_retrieved(retrieved)

        return retrieved[:self.config.retrieval_top_k]

    def _rerank_retrieved(self, retrieved: List[Dict]) -> List[Dict]:
        """Rerank retrieved knowledge by relevance and recency"""
        scored = []

        for item in retrieved:
            score = item['confidence']

            # Apply recency bias
            if 'timestamp' in item and item['timestamp']:
                age_hours = (datetime.utcnow() - item['timestamp']).total_seconds() / 3600
                recency_factor = np.exp(-age_hours / 24)  # Decay over 24 hours
                score += self.config.recency_bias * recency_factor

            # Boost frequently accessed knowledge
            if 'access_count' in item:
                access_boost = min(0.1, item['access_count'] / 100)
                score += access_boost

            scored.append((item, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [item for item, score in scored]

    def _build_augmented_context(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        retrieved_knowledge: List[Dict]
    ) -> str:
        """Build context combining conversation history and retrieved knowledge"""
        context_parts = []

        # Add retrieved knowledge first (as context)
        if retrieved_knowledge:
            context_parts.append("# Relevant Context:")
            for i, knowledge in enumerate(retrieved_knowledge[:3], 1):
                content = knowledge['content']
                source = knowledge['source']
                context_parts.append(f"{i}. [{source}] {content}")
            context_parts.append("")

        # Add conversation history
        if conversation_history:
            context_parts.append("# Conversation History:")
            for turn in conversation_history[-5:]:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                context_parts.append(f"{role.capitalize()}: {content}")
            context_parts.append("")

        # Add current input
        context_parts.append("# Current Question:")
        context_parts.append(f"User: {user_input}")
        context_parts.append("Bot:")

        return "\n".join(context_parts)

    def _generate_augmented_response(
        self,
        user_input: str,
        augmented_context: str,
        retrieved_knowledge: List[Dict]
    ) -> Dict[str, Any]:
        """Generate response using augmented context"""
        # Tokenize context
        inputs = self.chatbot.tokenizer(
            augmented_context,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_context_tokens
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.chatbot.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config.max_context_tokens + 200,
                temperature=self.chatbot.config.temperature,
                top_p=self.chatbot.config.top_p,
                top_k=self.chatbot.config.top_k,
                repetition_penalty=self.chatbot.config.repetition_penalty,
                pad_token_id=self.chatbot.tokenizer.pad_token_id,
                do_sample=True,
                num_return_sequences=1
            )

        # Decode
        response = self.chatbot.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Calculate confidence (simplified)
        confidence = self._calculate_response_confidence(
            response,
            retrieved_knowledge
        )

        return {
            'response': response,
            'confidence': confidence,
            'context_used': augmented_context,
            'retrieved_count': len(retrieved_knowledge)
        }

    def _calculate_response_confidence(
        self,
        response: str,
        retrieved_knowledge: List[Dict]
    ) -> float:
        """Calculate confidence in the generated response"""
        # Base confidence from neural model
        base_confidence = 0.5

        # Boost confidence if retrieved knowledge was used
        if retrieved_knowledge:
            avg_knowledge_confidence = np.mean([
                k['confidence'] for k in retrieved_knowledge
            ])
            # Weighted combination
            confidence = (
                (1 - self.config.knowledge_weight) * base_confidence +
                self.config.knowledge_weight * avg_knowledge_confidence
            )
        else:
            confidence = base_confidence

        # Adjust based on response length and coherence
        response_length = len(response.split())
        if response_length < 5:
            confidence *= 0.7  # Very short responses are less confident
        elif response_length > 100:
            confidence *= 0.9  # Very long responses might be rambling

        return min(1.0, confidence)

    def _postprocess_response(
        self,
        response_data: Dict[str, Any],
        retrieved_knowledge: List[Dict],
        query_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """Post-process response and add metadata"""
        response = response_data['response']
        confidence = response_data['confidence']

        # Extract sources
        sources = []
        for knowledge in retrieved_knowledge:
            if knowledge['confidence'] >= self.config.min_relevance_score:
                sources.append({
                    'source': knowledge['source'],
                    'confidence': knowledge['confidence'],
                    'content_preview': knowledge['content'][:100] + '...'
                })

        # Calculate semantic similarity between response and query
        response_embedding = self.chatbot.encode_text(response)
        semantic_similarity = float(np.dot(query_embedding, response_embedding))

        return {
            'response': response,
            'confidence': confidence,
            'sources': sources,
            'retrieved_knowledge_count': len(retrieved_knowledge),
            'semantic_similarity': semantic_similarity,
            'timestamp': datetime.utcnow().isoformat(),
            'rag_enhanced': True
        }

    def _store_interaction(
        self,
        user_input: str,
        response: str,
        embedding: np.ndarray,
        confidence: float,
        user_id: Optional[str]
    ):
        """Store the interaction for future retrieval"""
        # Generate conversation ID
        conv_id = f"conv_{datetime.utcnow().timestamp()}"

        # Add to knowledge graph
        self.knowledge_manager.add_knowledge_from_conversation(
            conversation_id=conv_id,
            user_input=user_input,
            bot_response=response,
            embedding=embedding,
            confidence=confidence
        )

        # Save to conversations table
        self.knowledge_manager.store.save_conversation(
            conversation_id=conv_id,
            user_input=user_input,
            bot_response=response
        )

        # Update user profile if provided
        if user_id:
            self.knowledge_manager.store.update_user_profile(user_id)

        logger.debug(f"Stored interaction: {conv_id}")

    def batch_learn_from_conversations(self, batch_size: int = 10):
        """Learn from unlearned conversations in batch"""
        # Get unlearned conversations
        conversations = self.knowledge_manager.store.get_conversations(
            limit=batch_size,
            learned_only=False
        )

        unlearned = [c for c in conversations if not c['learned']]

        if not unlearned:
            logger.info("No new conversations to learn from")
            return

        logger.info(f"Learning from {len(unlearned)} conversations")

        # Add to experience buffer and trigger learning
        for conv in unlearned:
            self.chatbot.learn_from_interaction(
                user_input=conv['user_input'],
                bot_response=conv['bot_response'],
                user_feedback=conv.get('user_feedback'),
                rating=conv.get('rating')
            )

            # Mark as learned
            self.knowledge_manager.store.mark_conversation_learned(conv['id'])

        # Perform continual learning steps
        for _ in range(len(unlearned) // 4 + 1):
            self.chatbot.continual_learning_step()

        logger.info("Batch learning completed")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        chatbot_stats = self.chatbot.get_stats()
        knowledge_stats = self.knowledge_manager.get_statistics()

        return {
            'chatbot': chatbot_stats,
            'knowledge_store': knowledge_stats,
            'rag_config': {
                'retrieval_top_k': self.config.retrieval_top_k,
                'knowledge_weight': self.config.knowledge_weight,
                'min_relevance_score': self.config.min_relevance_score
            }
        }


def create_rag_system(
    db_path: str = "knowledge_store.db",
    chatbot_config: Optional[ChatbotConfig] = None,
    rag_config: Optional[RAGConfig] = None
) -> RetrievalAugmentedGenerator:
    """Create a complete RAG system"""
    from neural_chatbot import create_chatbot

    # Create components
    chatbot = create_chatbot(chatbot_config)
    knowledge_manager = KnowledgeGraphManager(db_path)

    # Create RAG system
    rag_system = RetrievalAugmentedGenerator(
        chatbot=chatbot,
        knowledge_manager=knowledge_manager,
        config=rag_config
    )

    logger.info("RAG system created successfully")

    return rag_system


if __name__ == '__main__':
    print("=" * 60)
    print("RAG SYSTEM TEST")
    print("=" * 60)

    # Create RAG system
    rag = create_rag_system(db_path="test_rag_knowledge.db")

    # Test conversation
    print("\n" + "-" * 60)
    print("TEST CONVERSATION")
    print("-" * 60)

    user_input = "What is machine learning?"
    print(f"\nUser: {user_input}")

    result = rag.generate_with_retrieval(
        user_input,
        user_id="test_user"
    )

    print(f"\nBot: {result['response']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Sources used: {result['retrieved_knowledge_count']}")
    print(f"RAG enhanced: {result['rag_enhanced']}")

    # Show statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    stats = rag.get_comprehensive_stats()
    print(json.dumps(stats, indent=2, default=str))

    print("\n✓ RAG system test complete!")
