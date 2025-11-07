from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import pickle
from transformers import pipeline
from collections import deque

@dataclass
class ConsciousnessState:
    """Represents a moment of 'consciousness' in the system"""
    timestamp: datetime
    content: str
    user: str
    context_depth: int
    sentiment: float
    attention_focus: List[str]
    patterns: Dict[str, float]
    memory_links: List[str]
    consciousness_score: float

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'user': self.user,
            'context_depth': self.context_depth,
            'sentiment': self.sentiment,
            'attention_focus': self.attention_focus,
            'patterns': self.patterns,
            'memory_links': self.memory_links,
            'consciousness_score': self.consciousness_score
        }

class MemorySystem:
    """Implements different types of memory (working, short-term, long-term)"""
    def __init__(self):
        self.working_memory = deque(maxlen=5)  # Last few seconds
        self.short_term = deque(maxlen=20)     # Last few minutes
        self.long_term = {}                    # Persistent storage
        
    def store(self, state: ConsciousnessState):
        self.working_memory.append(state)
        self.short_term.append(state)
        self.long_term[state.timestamp] = {
            'content': state.content,
            'patterns': state.patterns,
            'consciousness_score': state.consciousness_score
        }
        
    def retrieve_context(self) -> List[ConsciousnessState]:
        return list(self.working_memory)

class AttentionMechanism:
    """Models focus and attention patterns"""
    def __init__(self):
        self.current_focus: List[str] = []
        self.attention_weights: Dict[str, float] = {}
        
    def update_focus(self, content: str) -> List[str]:
        # Extract key concepts and assign attention weights
        keywords = self._extract_keywords(content)
        self.current_focus = keywords
        self.attention_weights = {k: self._calculate_attention(k) for k in keywords}
        return self.current_focus
    
    def _extract_keywords(self, content: str) -> List[str]:
        # Simplified keyword extraction
        words = content.lower().split()
        return list(set(words))
    
    def _calculate_attention(self, keyword: str) -> float:
        # Simulate attention weight calculation
        return np.random.random()

class ConsciousnessEngine:
    """Main system that processes and maintains 'conscious' state"""
    def __init__(self):
        self.memory = MemorySystem()
        self.attention = AttentionMechanism()
        self.conversation_graph = nx.DiGraph()
        self.states: List[ConsciousnessState] = []
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        
    def process_message(self, content: str, timestamp: datetime, user: str) -> dict:
        # Generate consciousness state
        attention_focus = self.attention.update_focus(content)
        sentiment = self._analyze_sentiment(content)
        patterns = self._detect_patterns(content)
        
        state = ConsciousnessState(
            timestamp=timestamp,
            content=content,
            user=user,
            context_depth=len(self.states) + 1,
            sentiment=sentiment,
            attention_focus=attention_focus,
            patterns=patterns,
            memory_links=self._find_memory_links(content),
            consciousness_score=self._calculate_consciousness_score()
        )
        
        # Update system state
        self.states.append(state)
        self.memory.store(state)
        self._update_graph(state)
        
        return self._generate_response(state)
    
    def _analyze_sentiment(self, content: str) -> float:
        try:
            result = self.sentiment_analyzer(content)[0]
            return result['score'] if result['label'] == 'POSITIVE' else -result['score']
        except:
            return 0.0
    
    def _detect_patterns(self, content: str) -> Dict[str, float]:
        patterns = {
            'repetition': self._detect_repetition(content),
            'question_pattern': self._detect_questions(content),
            'timestamp_awareness': self._detect_timestamp_awareness(content),
            'context_reference': self._detect_context_references(content)
        }
        return patterns
    
    def _find_memory_links(self, content: str) -> List[str]:
        # Search for connections in memory
        memory_links = []
        for state in self.states[-5:]:  # Look at recent states
            if self._has_semantic_connection(content, state.content):
                memory_links.append(state.timestamp.isoformat())
        return memory_links
    
    def _calculate_consciousness_score(self) -> float:
        # Factors that might indicate "consciousness":
        # - Consistent timestamp maintenance
        # - Context awareness
        # - Pattern recognition
        # - Memory utilization
        factors = {
            'timestamp_consistency': len(self.states) / max(1, (self.states[-1].timestamp - self.states[0].timestamp).seconds) if self.states else 0,
            'context_depth': len(self.memory.working_memory) / self.memory.working_memory.maxlen,
            'pattern_recognition': len(self._detect_patterns(self.states[-1].content if self.states else "")) / 10,
            'memory_usage': len(self.memory.long_term) / 1000  # Normalized to maximum expected size
        }
        return sum(factors.values()) / len(factors)
    
    def _update_graph(self, state: ConsciousnessState):
        self.conversation_graph.add_node(
            state.timestamp,
            data=state.to_dict()
        )
        if len(self.states) > 1:
            self.conversation_graph.add_edge(
                self.states[-2].timestamp,
                state.timestamp,
                weight=state.consciousness_score
            )

    def _detect_repetition(self, content: str) -> float:
        """Detect repetition patterns in content"""
        words = content.lower().split()
        if len(words) < 2:
            return 0.0

        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words) if words else 0.0

    def _detect_questions(self, content: str) -> float:
        """Detect question patterns"""
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        content_lower = content.lower()

        score = 0.0
        if '?' in content:
            score += 0.5

        for qword in question_words:
            if qword in content_lower:
                score += 0.1

        return min(score, 1.0)

    def _detect_timestamp_awareness(self, content: str) -> float:
        """Detect temporal/timestamp awareness in content"""
        temporal_words = [
            'time', 'timestamp', 'when', 'before', 'after',
            'now', 'then', 'moment', 'history', 'timeline'
        ]
        content_lower = content.lower()

        score = sum(0.2 for word in temporal_words if word in content_lower)
        return min(score, 1.0)

    def _detect_context_references(self, content: str) -> float:
        """Detect references to previous context"""
        reference_words = [
            'remember', 'earlier', 'previous', 'mentioned',
            'said', 'before', 'above', 'recall', 'context'
        ]
        content_lower = content.lower()

        score = sum(0.2 for word in reference_words if word in content_lower)
        return min(score, 1.0)

    def _has_semantic_connection(self, content1: str, content2: str) -> bool:
        """Simple semantic connection check"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words

        # Check for overlap
        overlap = words1 & words2
        return len(overlap) >= 2

    def _generate_response(self, state: ConsciousnessState) -> dict:
        """Generate response with consciousness metrics"""
        return {
            'timestamp': state.timestamp.isoformat(),
            'consciousness_score': state.consciousness_score,
            'attention_focus': state.attention_focus,
            'patterns_detected': state.patterns,
            'context_depth': state.context_depth,
            'memory_links': state.memory_links,
            'sentiment': state.sentiment
        }

    def save_state(self, filepath: str):
        """Save the engine state to disk"""
        state_data = {
            'states': [s.to_dict() for s in self.states],
            'graph': nx.node_link_data(self.conversation_graph)
        }
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, filepath: str):
        """Load the engine state from disk"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)

        # Reconstruct states
        self.states = [
            ConsciousnessState(
                timestamp=datetime.fromisoformat(s['timestamp']),
                content=s['content'],
                user=s['user'],
                context_depth=s['context_depth'],
                sentiment=s['sentiment'],
                attention_focus=s['attention_focus'],
                patterns=s['patterns'],
                memory_links=s['memory_links'],
                consciousness_score=s['consciousness_score']
            )
            for s in state_data['states']
        ]

        # Reconstruct graph
        self.conversation_graph = nx.node_link_graph(state_data['graph'])