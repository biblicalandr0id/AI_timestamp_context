"""
Advanced Semantic Analysis Engine
Provides embedding-based similarity, semantic search, and topic modeling
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import json


@dataclass
class SemanticCluster:
    """Represents a semantic cluster of related messages"""
    cluster_id: int
    centroid: np.ndarray
    messages: List[str]
    timestamps: List[datetime]
    coherence_score: float
    keywords: List[str]


class EmbeddingEngine:
    """Handles text embeddings and semantic similarity"""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.cache: Dict[str, np.ndarray] = {}
        self.model_loaded = False
        self._model = None

    def _lazy_load_model(self):
        """Lazy load the embedding model only when needed"""
        if not self.model_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_loaded = True
            except ImportError:
                print("Warning: sentence-transformers not installed. Using fallback embeddings.")
                self.model_loaded = False

    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text
            use_cache: Whether to use cached embeddings

        Returns:
            Embedding vector
        """
        if use_cache and text in self.cache:
            return self.cache[text]

        self._lazy_load_model()

        if self.model_loaded and self._model:
            embedding = self._model.encode(text, convert_to_numpy=True)
        else:
            # Fallback: Simple hash-based embedding
            embedding = self._fallback_embedding(text)

        if use_cache:
            self.cache[text] = embedding

        return embedding

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding based on character and word statistics"""
        words = text.lower().split()

        # Create features
        features = []

        # Length features
        features.append(len(text) / 1000.0)  # Normalized length
        features.append(len(words) / 100.0)  # Normalized word count
        features.append(np.mean([len(w) for w in words]) if words else 0)

        # Character distribution
        char_counts = defaultdict(int)
        for char in text.lower():
            if char.isalpha():
                char_counts[char] += 1

        for char in 'abcdefghijklmnopqrstuvwxyz':
            features.append(char_counts[char] / max(len(text), 1))

        # Word-level features
        unique_words = len(set(words))
        features.append(unique_words / max(len(words), 1))  # Lexical diversity

        # Punctuation features
        features.append(text.count('?') / max(len(text), 1))
        features.append(text.count('!') / max(len(text), 1))
        features.append(text.count('.') / max(len(text), 1))

        # Pad to embedding dimension
        embedding = np.array(features + [0.0] * (self.embedding_dim - len(features)))
        return embedding[:self.embedding_dim]

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def batch_similarity(self, query: str, texts: List[str]) -> List[float]:
        """Calculate similarity between query and multiple texts"""
        query_emb = self.encode(query)
        similarities = []

        for text in texts:
            text_emb = self.encode(text)
            sim = self.cosine_similarity(query_emb, text_emb)
            similarities.append(sim)

        return similarities


class TopicModeler:
    """Simple topic modeling using clustering"""

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.topics: List[SemanticCluster] = []

    def extract_topics(
        self,
        texts: List[str],
        timestamps: Optional[List[datetime]] = None,
        n_topics: int = 5
    ) -> List[SemanticCluster]:
        """
        Extract topics from a collection of texts

        Args:
            texts: List of text documents
            timestamps: Optional timestamps for each text
            n_topics: Number of topics to extract

        Returns:
            List of semantic clusters representing topics
        """
        if not texts:
            return []

        if timestamps is None:
            timestamps = [datetime.utcnow()] * len(texts)

        # Generate embeddings
        embeddings = np.array([self.embedding_engine.encode(text) for text in texts])

        # Simple k-means clustering
        clusters = self._simple_kmeans(embeddings, n_topics)

        # Create semantic clusters
        topics = []
        for cluster_id in range(n_topics):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]

            if not cluster_indices:
                continue

            cluster_texts = [texts[i] for i in cluster_indices]
            cluster_timestamps = [timestamps[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Calculate coherence
            coherence = self._calculate_coherence(cluster_embeddings, centroid)

            # Extract keywords
            keywords = self._extract_keywords(cluster_texts)

            topic = SemanticCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                messages=cluster_texts,
                timestamps=cluster_timestamps,
                coherence_score=coherence,
                keywords=keywords
            )
            topics.append(topic)

        self.topics = topics
        return topics

    def _simple_kmeans(self, embeddings: np.ndarray, k: int, max_iters: int = 100) -> List[int]:
        """Simple k-means clustering implementation"""
        n_samples = len(embeddings)

        # Initialize centroids randomly
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = embeddings[indices].copy()

        assignments = [0] * n_samples

        for _ in range(max_iters):
            # Assign points to nearest centroid
            new_assignments = []
            for emb in embeddings:
                distances = [np.linalg.norm(emb - centroid) for centroid in centroids]
                new_assignments.append(int(np.argmin(distances)))

            # Check convergence
            if new_assignments == assignments:
                break

            assignments = new_assignments

            # Update centroids
            for cluster_id in range(k):
                cluster_points = [embeddings[i] for i, c in enumerate(assignments) if c == cluster_id]
                if cluster_points:
                    centroids[cluster_id] = np.mean(cluster_points, axis=0)

        return assignments

    def _calculate_coherence(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
        """Calculate cluster coherence score"""
        if len(embeddings) == 0:
            return 0.0

        distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
        avg_distance = np.mean(distances)

        # Convert to similarity score (inverse of distance)
        coherence = 1.0 / (1.0 + avg_distance)
        return float(coherence)

    def _extract_keywords(self, texts: List[str], top_n: int = 5) -> List[str]:
        """Extract keywords from cluster texts"""
        # Simple word frequency approach
        word_freq = defaultdict(int)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'can',
                      'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from'}

        for text in texts:
            words = text.lower().split()
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if word and len(word) > 2 and word not in stop_words:
                    word_freq[word] += 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:top_n]]

        return keywords


class SemanticSearchEngine:
    """Advanced semantic search capabilities"""

    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.indexed_texts: List[str] = []
        self.indexed_embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def index(self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Index texts for semantic search"""
        self.indexed_texts = texts
        self.indexed_embeddings = [self.embedding_engine.encode(text) for text in texts]

        if metadata is None:
            self.metadata = [{} for _ in texts]
        else:
            self.metadata = metadata

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for semantically similar texts

        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (text, similarity_score, metadata) tuples
        """
        if not self.indexed_texts:
            return []

        query_emb = self.embedding_engine.encode(query)

        # Calculate similarities
        results = []
        for i, text_emb in enumerate(self.indexed_embeddings):
            similarity = self.embedding_engine.cosine_similarity(query_emb, text_emb)

            if similarity >= min_similarity:
                results.append((
                    self.indexed_texts[i],
                    similarity,
                    self.metadata[i]
                ))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar texts to a given text"""
        results = self.search(text, top_k=top_k + 1 if exclude_self else top_k)

        if exclude_self:
            # Remove the text itself if it's in the index
            results = [(t, s, m) for t, s, m in results if t != text][:top_k]

        return results


class AdvancedSemanticEngine:
    """
    Unified semantic analysis engine combining embeddings,
    topic modeling, and semantic search
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_engine = EmbeddingEngine(embedding_dim)
        self.topic_modeler = TopicModeler(self.embedding_engine)
        self.search_engine = SemanticSearchEngine(self.embedding_engine)

    def analyze_conversation(
        self,
        messages: List[str],
        timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis on a conversation

        Args:
            messages: List of message texts
            timestamps: Optional timestamps for messages

        Returns:
            Analysis results including topics, clusters, and insights
        """
        if not messages:
            return {
                'message_count': 0,
                'topics': [],
                'semantic_density': 0.0,
                'conversation_flow': []
            }

        # Extract topics
        n_topics = min(5, max(1, len(messages) // 3))
        topics = self.topic_modeler.extract_topics(messages, timestamps, n_topics)

        # Calculate semantic density (avg pairwise similarity)
        semantic_density = self._calculate_semantic_density(messages)

        # Analyze conversation flow
        flow = self._analyze_conversation_flow(messages, timestamps)

        # Index for search
        metadata = [{'index': i, 'timestamp': timestamps[i] if timestamps else None}
                    for i in range(len(messages))]
        self.search_engine.index(messages, metadata)

        return {
            'message_count': len(messages),
            'topics': [
                {
                    'cluster_id': t.cluster_id,
                    'size': len(t.messages),
                    'coherence': t.coherence_score,
                    'keywords': t.keywords,
                    'timestamp_range': (min(t.timestamps), max(t.timestamps)) if t.timestamps else None
                }
                for t in topics
            ],
            'semantic_density': semantic_density,
            'conversation_flow': flow
        }

    def _calculate_semantic_density(self, messages: List[str]) -> float:
        """Calculate average semantic similarity between consecutive messages"""
        if len(messages) < 2:
            return 1.0

        similarities = []
        for i in range(len(messages) - 1):
            emb1 = self.embedding_engine.encode(messages[i])
            emb2 = self.embedding_engine.encode(messages[i + 1])
            sim = self.embedding_engine.cosine_similarity(emb1, emb2)
            similarities.append(sim)

        return float(np.mean(similarities))

    def _analyze_conversation_flow(
        self,
        messages: List[str],
        timestamps: Optional[List[datetime]]
    ) -> List[Dict[str, Any]]:
        """Analyze the flow and progression of conversation"""
        flow = []

        for i, msg in enumerate(messages):
            if i == 0:
                flow.append({
                    'index': i,
                    'message': msg[:50] + '...' if len(msg) > 50 else msg,
                    'timestamp': timestamps[i] if timestamps else None,
                    'transition_type': 'start',
                    'semantic_shift': 0.0
                })
            else:
                # Calculate semantic shift
                emb_prev = self.embedding_engine.encode(messages[i - 1])
                emb_curr = self.embedding_engine.encode(msg)
                similarity = self.embedding_engine.cosine_similarity(emb_prev, emb_curr)

                # Classify transition
                if similarity > 0.7:
                    transition = 'continuation'
                elif similarity > 0.4:
                    transition = 'related_shift'
                else:
                    transition = 'topic_change'

                flow.append({
                    'index': i,
                    'message': msg[:50] + '...' if len(msg) > 50 else msg,
                    'timestamp': timestamps[i] if timestamps else None,
                    'transition_type': transition,
                    'semantic_shift': 1.0 - similarity
                })

        return flow

    def save_analysis(self, filepath: str, analysis: Dict[str, Any]):
        """Save analysis results to JSON file"""
        # Convert datetime objects to strings
        serializable = json.loads(
            json.dumps(analysis, default=str)
        )

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

    def load_analysis(self, filepath: str) -> Dict[str, Any]:
        """Load analysis results from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
