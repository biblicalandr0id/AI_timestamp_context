"""
Comprehensive Analytics Dashboard
Tracks and visualizes chatbot performance, learning metrics, and usage patterns
Real-time monitoring and historical analysis
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from pathlib import Path
import sqlite3

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("plotly not available. Install with: pip install plotly")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("pandas not available. Install with: pip install pandas")


class MetricsCollector:
    """Collect and store chatbot metrics"""

    def __init__(self, db_path: str = "metrics.db"):
        """
        Initialize metrics collector

        Args:
            db_path: Path to metrics database
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # Real-time metrics (in-memory)
        self.realtime_metrics = {
            'response_times': deque(maxlen=100),
            'confidence_scores': deque(maxlen=100),
            'active_users': set(),
            'requests_per_minute': deque(maxlen=60),
            'error_count': 0,
            'success_count': 0
        }

    def _init_db(self):
        """Initialize metrics database"""
        cursor = self.conn.cursor()

        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                duration_seconds REAL,
                message_count INTEGER,
                avg_confidence REAL,
                feedback_score INTEGER
            )
        """)

        # Responses table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                timestamp TEXT,
                user_message TEXT,
                bot_response TEXT,
                confidence REAL,
                response_time_ms REAL,
                rag_used BOOLEAN,
                knowledge_sources INTEGER,
                feedback INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)

        # Learning metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                training_loss REAL,
                validation_loss REAL,
                knowledge_items INTEGER,
                episodic_memories INTEGER,
                semantic_memories INTEGER,
                experience_buffer_size INTEGER
            )
        """)

        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cpu_percent REAL,
                memory_mb REAL,
                active_connections INTEGER,
                requests_per_second REAL,
                error_rate REAL
            )
        """)

        self.conn.commit()

    def log_conversation(self, user_id: str, duration: float, message_count: int,
                        avg_confidence: float, feedback: Optional[int] = None) -> int:
        """Log a conversation"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (user_id, timestamp, duration_seconds, message_count, avg_confidence, feedback_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, datetime.now().isoformat(), duration, message_count, avg_confidence, feedback))

        self.conn.commit()
        return cursor.lastrowid

    def log_response(self, conversation_id: int, user_message: str, bot_response: str,
                    confidence: float, response_time: float, rag_used: bool,
                    knowledge_sources: int, feedback: Optional[int] = None):
        """Log a single response"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO responses (conversation_id, timestamp, user_message, bot_response,
                                  confidence, response_time_ms, rag_used, knowledge_sources, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (conversation_id, datetime.now().isoformat(), user_message, bot_response,
              confidence, response_time, rag_used, knowledge_sources, feedback))

        self.conn.commit()

        # Update real-time metrics
        self.realtime_metrics['response_times'].append(response_time)
        self.realtime_metrics['confidence_scores'].append(confidence)
        self.realtime_metrics['success_count'] += 1

    def log_learning_metrics(self, training_loss: float, validation_loss: float,
                            knowledge_items: int, episodic: int, semantic: int, buffer_size: int):
        """Log learning/training metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO learning_metrics (timestamp, training_loss, validation_loss,
                                         knowledge_items, episodic_memories, semantic_memories, experience_buffer_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), training_loss, validation_loss,
              knowledge_items, episodic, semantic, buffer_size))

        self.conn.commit()

    def log_system_metrics(self, cpu: float, memory: float, connections: int, rps: float, error_rate: float):
        """Log system metrics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO system_metrics (timestamp, cpu_percent, memory_mb, active_connections, requests_per_second, error_rate)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), cpu, memory, connections, rps, error_rate))

        self.conn.commit()

    def get_response_metrics(self, hours: int = 24) -> List[Dict]:
        """Get response metrics for last N hours"""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute("""
            SELECT timestamp, confidence, response_time_ms, rag_used, feedback
            FROM responses
            WHERE timestamp >= ?
            ORDER BY timestamp
        """, (since,))

        columns = ['timestamp', 'confidence', 'response_time', 'rag_used', 'feedback']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_learning_history(self, days: int = 7) -> List[Dict]:
        """Get learning metrics history"""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT * FROM learning_metrics
            WHERE timestamp >= ?
            ORDER BY timestamp
        """, (since,))

        columns = ['id', 'timestamp', 'training_loss', 'validation_loss',
                  'knowledge_items', 'episodic_memories', 'semantic_memories', 'experience_buffer_size']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_summary_statistics(self) -> Dict:
        """Get overall summary statistics"""
        cursor = self.conn.cursor()

        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]

        # Total responses
        cursor.execute("SELECT COUNT(*) FROM responses")
        total_responses = cursor.fetchone()[0]

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM responses")
        avg_confidence = cursor.fetchone()[0] or 0

        # Average response time
        cursor.execute("SELECT AVG(response_time_ms) FROM responses")
        avg_response_time = cursor.fetchone()[0] or 0

        # Feedback distribution
        cursor.execute("SELECT feedback, COUNT(*) FROM responses WHERE feedback IS NOT NULL GROUP BY feedback")
        feedback_dist = dict(cursor.fetchall())

        # Recent activity (last 24h)
        since_24h = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM responses WHERE timestamp >= ?", (since_24h,))
        responses_24h = cursor.fetchone()[0]

        return {
            'total_conversations': total_conversations,
            'total_responses': total_responses,
            'avg_confidence': avg_confidence,
            'avg_response_time_ms': avg_response_time,
            'feedback_distribution': feedback_dist,
            'responses_last_24h': responses_24h,
            'realtime': {
                'success_rate': self.realtime_metrics['success_count'] / max(1, self.realtime_metrics['success_count'] + self.realtime_metrics['error_count']),
                'active_users': len(self.realtime_metrics['active_users']),
                'avg_response_time_recent': sum(self.realtime_metrics['response_times']) / max(1, len(self.realtime_metrics['response_times'])),
                'avg_confidence_recent': sum(self.realtime_metrics['confidence_scores']) / max(1, len(self.realtime_metrics['confidence_scores']))
            }
        }


class AnalyticsDashboard:
    """Generate comprehensive analytics dashboard"""

    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize dashboard

        Args:
            metrics_collector: MetricsCollector instance
        """
        self.metrics = metrics_collector

    def generate_dashboard(self, output_path: str = "analytics_dashboard.html", hours=24):
        """
        Generate complete analytics dashboard

        Args:
            output_path: Output HTML file path
            hours: Hours of history to show
        """
        if not PLOTLY_AVAILABLE:
            print("plotly required!")
            return

        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Response Time Over Time',
                'Confidence Scores Distribution',
                'Responses per Hour',
                'Feedback Distribution',
                'Learning Progress',
                'RAG Usage Statistics',
                'Knowledge Growth',
                'System Performance'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'indicator'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )

        # Get data
        response_data = self.metrics.get_response_metrics(hours)
        learning_data = self.metrics.get_learning_history(days=7)
        summary = self.metrics.get_summary_statistics()

        # 1. Response time over time
        if response_data:
            timestamps = [r['timestamp'] for r in response_data]
            response_times = [r['response_time'] for r in response_data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=response_times, mode='lines+markers',
                          name='Response Time', line=dict(color='#2196F3')),
                row=1, col=1
            )

        # 2. Confidence distribution
        if response_data:
            confidences = [r['confidence'] for r in response_data]

            fig.add_trace(
                go.Histogram(x=confidences, nbinsx=20, name='Confidence',
                           marker=dict(color='#4CAF50')),
                row=1, col=2
            )

        # 3. Responses per hour
        if response_data:
            hourly_counts = defaultdict(int)
            for r in response_data:
                hour = datetime.fromisoformat(r['timestamp']).strftime('%Y-%m-%d %H:00')
                hourly_counts[hour] += 1

            hours_list = sorted(hourly_counts.keys())
            counts = [hourly_counts[h] for h in hours_list]

            fig.add_trace(
                go.Bar(x=hours_list, y=counts, name='Responses/Hour',
                      marker=dict(color='#FF9800')),
                row=2, col=1
            )

        # 4. Feedback distribution
        feedback_dist = summary['feedback_distribution']
        if feedback_dist:
            labels = ['Positive' if k == 1 else 'Negative' if k == -1 else 'Neutral' for k in feedback_dist.keys()]
            values = list(feedback_dist.values())

            fig.add_trace(
                go.Pie(labels=labels, values=values, name='Feedback'),
                row=2, col=2
            )

        # 5. Learning progress
        if learning_data:
            timestamps = [l['timestamp'] for l in learning_data]
            train_loss = [l['training_loss'] for l in learning_data]
            val_loss = [l['validation_loss'] for l in learning_data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=train_loss, mode='lines',
                          name='Training Loss', line=dict(color='#F44336')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=timestamps, y=val_loss, mode='lines',
                          name='Validation Loss', line=dict(color='#9C27B0')),
                row=3, col=1
            )

        # 6. RAG usage
        if response_data:
            rag_used = sum(1 for r in response_data if r['rag_used'])
            rag_not_used = len(response_data) - rag_used

            fig.add_trace(
                go.Bar(x=['RAG Used', 'RAG Not Used'], y=[rag_used, rag_not_used],
                      name='RAG Usage', marker=dict(color=['#4CAF50', '#FF5722'])),
                row=3, col=2
            )

        # 7. Knowledge growth
        if learning_data:
            timestamps = [l['timestamp'] for l in learning_data]
            knowledge = [l['knowledge_items'] for l in learning_data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=knowledge, mode='lines+markers',
                          name='Knowledge Items', line=dict(color='#00BCD4'),
                          fill='tozeroy'),
                row=4, col=1
            )

        # 8. Overall performance indicator
        avg_confidence_pct = summary['avg_confidence'] * 100

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_confidence_pct,
                title={'text': "Avg Confidence %"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFCDD2"},
                        {'range': [50, 75], 'color': "#FFF9C4"},
                        {'range': [75, 100], 'color': "#C8E6C9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=4, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="🤖 AI Chatbot Analytics Dashboard",
            title_font_size=24,
            showlegend=True,
            height=1800,
            width=1400
        )

        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Response Time (ms)", row=1, col=1)

        fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_yaxes(title_text="Number of Responses", row=2, col=1)

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Loss", row=3, col=1)

        fig.update_xaxes(title_text="Date", row=4, col=1)
        fig.update_yaxes(title_text="Knowledge Items", row=4, col=1)

        # Save
        fig.write_html(output_path)
        print(f"✅ Analytics dashboard saved to {output_path}")

    def generate_realtime_dashboard(self, output_path: str = "realtime_dashboard.html"):
        """Generate real-time dashboard (simplified)"""
        if not PLOTLY_AVAILABLE:
            print("plotly required!")
            return

        summary = self.metrics.get_summary_statistics()
        realtime = summary['realtime']

        # Create figure with indicators
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'indicator'}, {'type': 'indicator'}]
            ],
            subplot_titles=('Success Rate', 'Active Users', 'Avg Response Time', 'Avg Confidence')
        )

        # Success rate
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=realtime['success_rate'] * 100,
                title={'text': "Success Rate %"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#4CAF50"}}
            ),
            row=1, col=1
        )

        # Active users
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=realtime['active_users'],
                title={'text': "Active Users"},
                delta={'reference': 10}
            ),
            row=1, col=2
        )

        # Avg response time
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=realtime['avg_response_time_recent'],
                title={'text': "Avg Response Time (ms)"},
                gauge={'axis': {'range': [0, 2000]}, 'bar': {'color': "#2196F3"}}
            ),
            row=2, col=1
        )

        # Avg confidence
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=realtime['avg_confidence_recent'] * 100,
                title={'text': "Avg Confidence %"},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#FF9800"}}
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text="⚡ Real-Time Dashboard",
            title_font_size=20,
            height=800,
            width=1000
        )

        fig.write_html(output_path)
        print(f"✅ Real-time dashboard saved to {output_path}")

    def export_report(self, output_path: str = "analytics_report.json"):
        """Export complete analytics report as JSON"""
        summary = self.metrics.get_summary_statistics()
        response_data = self.metrics.get_response_metrics(hours=24)
        learning_data = self.metrics.get_learning_history(days=7)

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'recent_responses': response_data[-10:],  # Last 10 responses
            'learning_history': learning_data,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Report exported to {output_path}")


def demo_analytics():
    """Demo the analytics dashboard"""
    print("Analytics Dashboard Demo")
    print("=" * 60)

    # Create collector
    collector = MetricsCollector("demo_metrics.db")

    # Simulate some data
    print("Simulating data...")
    import random

    # Log some responses
    conv_id = collector.log_conversation("user123", 120.5, 10, 0.85, 1)

    for i in range(50):
        collector.log_response(
            conv_id,
            f"User message {i}",
            f"Bot response {i}",
            confidence=random.uniform(0.6, 0.95),
            response_time=random.uniform(100, 500),
            rag_used=random.choice([True, False]),
            knowledge_sources=random.randint(0, 5),
            feedback=random.choice([1, -1, None])
        )

    # Log learning metrics
    for i in range(10):
        collector.log_learning_metrics(
            training_loss=2.0 - i * 0.1,
            validation_loss=2.2 - i * 0.1,
            knowledge_items=100 + i * 10,
            episodic=50 + i * 5,
            semantic=30 + i * 3,
            buffer_size=200 + i * 20
        )

    # Create dashboard
    dashboard = AnalyticsDashboard(collector)

    print("\n📊 Generating dashboards...")
    dashboard.generate_dashboard("demo_analytics.html")
    dashboard.generate_realtime_dashboard("demo_realtime.html")
    dashboard.export_report("demo_report.json")

    print("\n✅ Demo complete! Check the generated files.")


if __name__ == '__main__':
    demo_analytics()
