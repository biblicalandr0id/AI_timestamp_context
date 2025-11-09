"""
Knowledge Storage System with Vector Database
Persistent storage using SQLite + pgvector-style functionality
Free and scalable with option for Supabase upgrade
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """Represents a node in the knowledge graph"""
    id: str
    content: str
    embedding: np.ndarray
    node_type: str  # 'fact', 'conversation', 'concept', 'procedure'
    confidence: float
    source: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0


@dataclass
class KnowledgeEdge:
    """Represents a relationship between knowledge nodes"""
    id: str
    source_id: str
    target_id: str
    relationship_type: str  # 'related_to', 'follows', 'contradicts', 'supports'
    strength: float
    created_at: datetime


class VectorStore:
    """
    SQLite-based vector store with similarity search
    Can be upgraded to Supabase pgvector for production
    """

    def __init__(self, db_path: str = "knowledge_store.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Knowledge nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                node_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            )
        """)

        # Knowledge edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES knowledge_nodes(id),
                FOREIGN KEY (target_id) REFERENCES knowledge_nodes(id)
            )
        """)

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_input TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                user_feedback TEXT,
                rating REAL,
                context TEXT,
                timestamp TEXT NOT NULL,
                learned BOOLEAN DEFAULT 0
            )
        """)

        # User profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                conversation_count INTEGER DEFAULT 0,
                total_feedback_score REAL DEFAULT 0,
                last_interaction TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Create indices for faster search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_node_type
            ON knowledge_nodes(node_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence
            ON knowledge_nodes(confidence)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
            ON conversations(timestamp)
        """)

        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def add_knowledge_node(self, node: KnowledgeNode) -> bool:
        """Add a knowledge node to the store"""
        try:
            cursor = self.conn.cursor()

            # Serialize embedding
            embedding_blob = pickle.dumps(node.embedding)

            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_nodes
                (id, content, embedding, node_type, confidence, source,
                 metadata, created_at, updated_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.content,
                embedding_blob,
                node.node_type,
                node.confidence,
                node.source,
                json.dumps(node.metadata),
                node.created_at.isoformat(),
                node.updated_at.isoformat(),
                node.access_count
            ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding knowledge node: {e}")
            return False

    def get_knowledge_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve a knowledge node by ID"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM knowledge_nodes WHERE id = ?
        """, (node_id,))

        row = cursor.fetchone()

        if row:
            # Update access count
            cursor.execute("""
                UPDATE knowledge_nodes
                SET access_count = access_count + 1
                WHERE id = ?
            """, (node_id,))
            self.conn.commit()

            return self._row_to_knowledge_node(row)

        return None

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        node_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Tuple[KnowledgeNode, float]]:
        """
        Search for similar nodes using cosine similarity

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            node_type: Filter by node type
            min_confidence: Minimum confidence threshold

        Returns:
            List of (node, similarity_score) tuples
        """
        cursor = self.conn.cursor()

        # Build query
        query = "SELECT * FROM knowledge_nodes WHERE confidence >= ?"
        params = [min_confidence]

        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Compute similarities
        results = []
        for row in rows:
            node = self._row_to_knowledge_node(row)

            # Cosine similarity
            similarity = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )

            results.append((node, float(similarity)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def add_knowledge_edge(self, edge: KnowledgeEdge) -> bool:
        """Add a relationship between knowledge nodes"""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO knowledge_edges
                (id, source_id, target_id, relationship_type, strength, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.relationship_type,
                edge.strength,
                edge.created_at.isoformat()
            ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding knowledge edge: {e}")
            return False

    def get_related_nodes(
        self,
        node_id: str,
        relationship_type: Optional[str] = None,
        min_strength: float = 0.0
    ) -> List[Tuple[KnowledgeNode, KnowledgeEdge]]:
        """Get nodes related to a given node"""
        cursor = self.conn.cursor()

        # Build query
        query = """
            SELECT n.*, e.*
            FROM knowledge_nodes n
            JOIN knowledge_edges e ON n.id = e.target_id
            WHERE e.source_id = ? AND e.strength >= ?
        """
        params = [node_id, min_strength]

        if relationship_type:
            query += " AND e.relationship_type = ?"
            params.append(relationship_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # Split row into node and edge parts
            node_data = row[:10]  # First 10 columns are node data
            edge_data = row[10:]  # Rest are edge data

            node = self._row_to_knowledge_node(node_data)
            edge = self._row_to_knowledge_edge(edge_data)

            results.append((node, edge))

        return results

    def save_conversation(
        self,
        conversation_id: str,
        user_input: str,
        bot_response: str,
        user_feedback: Optional[str] = None,
        rating: Optional[float] = None,
        context: Optional[Dict] = None
    ) -> bool:
        """Save a conversation to the database"""
        try:
            cursor = self.conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO conversations
                (id, user_input, bot_response, user_feedback, rating, context, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                user_input,
                bot_response,
                user_feedback,
                rating,
                json.dumps(context) if context else None,
                datetime.utcnow().isoformat()
            ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    def get_conversations(
        self,
        limit: int = 100,
        learned_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve conversations"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM conversations"
        if learned_only:
            query += " WHERE learned = 1"
        query += " ORDER BY timestamp DESC LIMIT ?"

        cursor.execute(query, (limit,))
        rows = cursor.fetchall()

        conversations = []
        for row in rows:
            conversations.append({
                'id': row[0],
                'user_input': row[1],
                'bot_response': row[2],
                'user_feedback': row[3],
                'rating': row[4],
                'context': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6],
                'learned': bool(row[7])
            })

        return conversations

    def mark_conversation_learned(self, conversation_id: str) -> bool:
        """Mark a conversation as learned"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE conversations SET learned = 1 WHERE id = ?
            """, (conversation_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error marking conversation as learned: {e}")
            return False

    def update_user_profile(
        self,
        user_id: str,
        preferences: Optional[Dict] = None,
        feedback_score: Optional[float] = None
    ) -> bool:
        """Update or create user profile"""
        try:
            cursor = self.conn.cursor()

            # Check if user exists
            cursor.execute("""
                SELECT * FROM user_profiles WHERE user_id = ?
            """, (user_id,))

            exists = cursor.fetchone() is not None

            if exists:
                # Update existing
                updates = []
                params = []

                if preferences:
                    updates.append("preferences = ?")
                    params.append(json.dumps(preferences))

                updates.append("conversation_count = conversation_count + 1")

                if feedback_score is not None:
                    updates.append("total_feedback_score = total_feedback_score + ?")
                    params.append(feedback_score)

                updates.append("last_interaction = ?")
                params.append(datetime.utcnow().isoformat())

                params.append(user_id)

                cursor.execute(f"""
                    UPDATE user_profiles
                    SET {', '.join(updates)}
                    WHERE user_id = ?
                """, params)

            else:
                # Create new
                cursor.execute("""
                    INSERT INTO user_profiles
                    (user_id, preferences, conversation_count, total_feedback_score,
                     last_interaction, created_at)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(preferences) if preferences else None,
                    feedback_score or 0.0,
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat()
                ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT * FROM user_profiles WHERE user_id = ?
        """, (user_id,))

        row = cursor.fetchone()

        if row:
            return {
                'user_id': row[0],
                'preferences': json.loads(row[1]) if row[1] else {},
                'conversation_count': row[2],
                'total_feedback_score': row[3],
                'average_feedback': row[3] / max(row[2], 1),
                'last_interaction': row[4],
                'created_at': row[5]
            }

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge store statistics"""
        cursor = self.conn.cursor()

        # Count nodes by type
        cursor.execute("""
            SELECT node_type, COUNT(*), AVG(confidence)
            FROM knowledge_nodes
            GROUP BY node_type
        """)
        node_stats = {row[0]: {'count': row[1], 'avg_confidence': row[2]}
                      for row in cursor.fetchall()}

        # Count total items
        cursor.execute("SELECT COUNT(*) FROM knowledge_nodes")
        total_nodes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_edges")
        total_edges = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM conversations WHERE learned = 1")
        learned_conversations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM user_profiles")
        total_users = cursor.fetchone()[0]

        return {
            'total_knowledge_nodes': total_nodes,
            'total_knowledge_edges': total_edges,
            'total_conversations': total_conversations,
            'learned_conversations': learned_conversations,
            'total_users': total_users,
            'node_stats_by_type': node_stats
        }

    def _row_to_knowledge_node(self, row: tuple) -> KnowledgeNode:
        """Convert database row to KnowledgeNode"""
        return KnowledgeNode(
            id=row[0],
            content=row[1],
            embedding=pickle.loads(row[2]),
            node_type=row[3],
            confidence=row[4],
            source=row[5],
            metadata=json.loads(row[6]) if row[6] else {},
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            access_count=row[9]
        )

    def _row_to_knowledge_edge(self, row: tuple) -> KnowledgeEdge:
        """Convert database row to KnowledgeEdge"""
        return KnowledgeEdge(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relationship_type=row[3],
            strength=row[4],
            created_at=datetime.fromisoformat(row[5])
        )

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __del__(self):
        """Cleanup"""
        self.close()


class KnowledgeGraphManager:
    """High-level knowledge graph management"""

    def __init__(self, db_path: str = "knowledge_store.db"):
        self.store = VectorStore(db_path)

    def add_knowledge_from_conversation(
        self,
        conversation_id: str,
        user_input: str,
        bot_response: str,
        embedding: np.ndarray,
        confidence: float = 0.7
    ) -> str:
        """Add knowledge extracted from a conversation"""
        # Create knowledge node
        node = KnowledgeNode(
            id=f"conv_{conversation_id}",
            content=f"Q: {user_input}\nA: {bot_response}",
            embedding=embedding,
            node_type='conversation',
            confidence=confidence,
            source='conversation',
            metadata={
                'user_input': user_input,
                'bot_response': bot_response
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        self.store.add_knowledge_node(node)
        return node.id

    def retrieve_relevant_knowledge(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.5
    ) -> List[KnowledgeNode]:
        """Retrieve relevant knowledge for a query"""
        results = self.store.vector_search(
            query_embedding,
            top_k=top_k,
            min_confidence=min_confidence
        )

        return [node for node, score in results]

    def build_knowledge_connections(
        self,
        node_id: str,
        related_node_ids: List[str],
        relationship_type: str = 'related_to',
        strength: float = 0.5
    ):
        """Build connections between knowledge nodes"""
        for related_id in related_node_ids:
            edge = KnowledgeEdge(
                id=f"{node_id}_{related_id}",
                source_id=node_id,
                target_id=related_id,
                relationship_type=relationship_type,
                strength=strength,
                created_at=datetime.utcnow()
            )
            self.store.add_knowledge_edge(edge)

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return self.store.get_statistics()


if __name__ == '__main__':
    print("=" * 60)
    print("KNOWLEDGE STORAGE SYSTEM TEST")
    print("=" * 60)

    # Create knowledge store
    store = VectorStore("test_knowledge.db")

    # Test adding a knowledge node
    test_embedding = np.random.rand(384)
    node = KnowledgeNode(
        id="test_001",
        content="Artificial Intelligence is the simulation of human intelligence by machines.",
        embedding=test_embedding,
        node_type='fact',
        confidence=0.9,
        source='test',
        metadata={'category': 'AI'},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    store.add_knowledge_node(node)
    print("\n✓ Knowledge node added")

    # Test retrieval
    retrieved = store.get_knowledge_node("test_001")
    print(f"✓ Retrieved node: {retrieved.content[:50]}...")

    # Test vector search
    query_embedding = test_embedding + np.random.rand(384) * 0.1
    results = store.vector_search(query_embedding, top_k=5)
    print(f"✓ Vector search found {len(results)} results")

    # Test conversation storage
    store.save_conversation(
        "conv_001",
        "What is AI?",
        "AI is artificial intelligence.",
        rating=0.9
    )
    print("✓ Conversation saved")

    # Get statistics
    stats = store.get_statistics()
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        print(f"{key}: {value}")

    store.close()
    print("\n✓ Knowledge store test complete!")
