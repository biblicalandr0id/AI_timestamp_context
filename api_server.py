"""
REST API Server for AI Timestamp Context System
Provides HTTP endpoints for all system functionality
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from functools import wraps

# Import system components
from ai_consciousness_model import ConsciousnessEngine
from multi_agent_system import create_default_multi_agent_system
from emotion_tracker import EmotionTracker
from semantic_engine import AdvancedSemanticEngine
from export_utils import ConversationExporter, AnalyticsExporter


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global system instances
consciousness_engine = None
multi_agent_system = None
emotion_tracker = None
semantic_engine = None

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size


def require_api_key(f):
    """Decorator for API key authentication (optional)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        # Add your API key validation logic here
        # For now, we'll accept any request
        return f(*args, **kwargs)
    return decorated_function


def initialize_systems():
    """Initialize all system components"""
    global consciousness_engine, multi_agent_system, emotion_tracker, semantic_engine

    consciousness_engine = ConsciousnessEngine()
    multi_agent_system = create_default_multi_agent_system()
    emotion_tracker = EmotionTracker()
    semantic_engine = AdvancedSemanticEngine()


@app.route('/')
def index():
    """API index with available endpoints"""
    return jsonify({
        'name': 'AI Timestamp Context API',
        'version': '1.0.0',
        'endpoints': {
            'consciousness': {
                'POST /api/consciousness/process': 'Process message through consciousness engine',
                'GET /api/consciousness/stats': 'Get consciousness statistics',
                'GET /api/consciousness/export': 'Export consciousness data'
            },
            'multi-agent': {
                'POST /api/agents/message': 'Send message to agents',
                'POST /api/agents/dialogue': 'Run multi-agent dialogue',
                'GET /api/agents/list': 'List all agents',
                'GET /api/agents/stats': 'Get agent statistics'
            },
            'emotion': {
                'POST /api/emotion/analyze': 'Analyze emotion in text',
                'GET /api/emotion/summary': 'Get emotion summary',
                'GET /api/emotion/trajectory': 'Get emotional trajectory'
            },
            'semantic': {
                'POST /api/semantic/analyze': 'Semantic analysis of conversation',
                'POST /api/semantic/search': 'Semantic search',
                'POST /api/semantic/topics': 'Extract topics'
            },
            'export': {
                'POST /api/export/json': 'Export to JSON',
                'POST /api/export/csv': 'Export to CSV',
                'POST /api/export/html': 'Export to HTML',
                'POST /api/export/markdown': 'Export to Markdown'
            }
        }
    })


# Consciousness Engine Endpoints

@app.route('/api/consciousness/process', methods=['POST'])
def process_consciousness_message():
    """Process a message through the consciousness engine"""
    data = request.get_json()

    if not data or 'content' not in data:
        return jsonify({'error': 'Missing content field'}), 400

    content = data['content']
    user = data.get('user', 'user')
    timestamp_str = data.get('timestamp')

    if timestamp_str:
        timestamp = datetime.fromisoformat(timestamp_str)
    else:
        timestamp = datetime.utcnow()

    result = consciousness_engine.process_message(content, timestamp, user)

    return jsonify(result)


@app.route('/api/consciousness/stats', methods=['GET'])
def get_consciousness_stats():
    """Get consciousness engine statistics"""
    states = consciousness_engine.states

    if not states:
        return jsonify({'message': 'No data yet'})

    import numpy as np
    avg_consciousness = float(np.mean([s.consciousness_score for s in states]))
    avg_sentiment = float(np.mean([s.sentiment for s in states]))

    return jsonify({
        'total_messages': len(states),
        'average_consciousness': avg_consciousness,
        'average_sentiment': avg_sentiment,
        'working_memory_size': len(consciousness_engine.memory.working_memory),
        'long_term_memory_size': len(consciousness_engine.memory.long_term)
    })


@app.route('/api/consciousness/export', methods=['GET'])
def export_consciousness_data():
    """Export consciousness data"""
    format = request.args.get('format', 'json')
    filepath = f'/tmp/consciousness_export.{format}'

    states_data = [s.to_dict() for s in consciousness_engine.states]

    if format == 'json':
        ConversationExporter.to_json(states_data, filepath)
    elif format == 'csv':
        ConversationExporter.to_csv(states_data, filepath)
    else:
        return jsonify({'error': 'Unsupported format'}), 400

    return send_file(filepath, as_attachment=True)


# Multi-Agent Endpoints

@app.route('/api/agents/message', methods=['POST'])
def send_agent_message():
    """Send a message to agents"""
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message field'}), 400

    message = data['message']
    target_agent = data.get('target_agent')
    timestamp_str = data.get('timestamp')

    if timestamp_str:
        timestamp = datetime.fromisoformat(timestamp_str)
    else:
        timestamp = datetime.utcnow()

    responses = multi_agent_system.process_user_message(
        message, timestamp, target_agent
    )

    return jsonify({
        'responses': [
            {
                'agent_id': r.agent_id,
                'content': r.content,
                'timestamp': r.timestamp.isoformat(),
                'message_type': r.message_type,
                'metadata': r.metadata
            }
            for r in responses
        ]
    })


@app.route('/api/agents/dialogue', methods=['POST'])
def run_agent_dialogue():
    """Run a multi-agent dialogue"""
    data = request.get_json()

    if not data or 'initial_message' not in data:
        return jsonify({'error': 'Missing initial_message field'}), 400

    initial_message = data['initial_message']
    num_turns = data.get('num_turns', 5)

    dialogue = multi_agent_system.run_multi_agent_dialogue(
        initial_message, num_turns
    )

    return jsonify({
        'dialogue': [
            {
                'agent_id': msg.agent_id,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'message_type': msg.message_type
            }
            for msg in dialogue
        ]
    })


@app.route('/api/agents/list', methods=['GET'])
def list_agents():
    """List all agents"""
    agents_info = []

    for agent_id, agent in multi_agent_system.agents.items():
        agents_info.append({
            'agent_id': agent.profile.agent_id,
            'name': agent.profile.name,
            'personality': agent.profile.personality.value,
            'expertise': agent.profile.expertise_domains
        })

    return jsonify({'agents': agents_info})


@app.route('/api/agents/stats', methods=['GET'])
def get_agent_stats():
    """Get agent statistics"""
    summary = multi_agent_system.get_interaction_summary()
    return jsonify(summary)


# Emotion Tracking Endpoints

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """Analyze emotion in text"""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400

    text = data['text']
    timestamp_str = data.get('timestamp')

    if timestamp_str:
        timestamp = datetime.fromisoformat(timestamp_str)
    else:
        timestamp = datetime.utcnow()

    state = emotion_tracker.track(text, timestamp)

    return jsonify({
        'primary_emotion': state.primary_emotion.value,
        'intensity': state.intensity.value,
        'valence': state.valence,
        'arousal': state.arousal,
        'dominance': state.dominance,
        'confidence': state.confidence,
        'triggers': state.triggers,
        'emotion_vector': {k.value: v for k, v in state.emotion_vector.items()}
    })


@app.route('/api/emotion/summary', methods=['GET'])
def get_emotion_summary():
    """Get emotion tracking summary"""
    summary = emotion_tracker.get_emotion_summary()
    return jsonify(summary)


@app.route('/api/emotion/trajectory', methods=['GET'])
def get_emotion_trajectory():
    """Get emotional trajectory"""
    trajectory = emotion_tracker.get_emotional_trajectory()
    return jsonify(trajectory)


# Semantic Analysis Endpoints

@app.route('/api/semantic/analyze', methods=['POST'])
def semantic_analyze():
    """Perform semantic analysis on conversation"""
    data = request.get_json()

    if not data or 'messages' not in data:
        return jsonify({'error': 'Missing messages field'}), 400

    messages = data['messages']
    timestamps = data.get('timestamps')

    if timestamps:
        timestamps = [datetime.fromisoformat(ts) for ts in timestamps]

    analysis = semantic_engine.analyze_conversation(messages, timestamps)

    return jsonify(analysis)


@app.route('/api/semantic/search', methods=['POST'])
def semantic_search():
    """Semantic search in conversation"""
    data = request.get_json()

    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query field'}), 400

    query = data['query']
    top_k = data.get('top_k', 5)
    min_similarity = data.get('min_similarity', 0.0)

    results = semantic_engine.search_engine.search(query, top_k, min_similarity)

    return jsonify({
        'results': [
            {
                'text': text,
                'similarity': float(sim),
                'metadata': metadata
            }
            for text, sim, metadata in results
        ]
    })


@app.route('/api/semantic/topics', methods=['POST'])
def extract_topics():
    """Extract topics from messages"""
    data = request.get_json()

    if not data or 'messages' not in data:
        return jsonify({'error': 'Missing messages field'}), 400

    messages = data['messages']
    n_topics = data.get('n_topics', 5)

    topics = semantic_engine.topic_modeler.extract_topics(messages, n_topics=n_topics)

    return jsonify({
        'topics': [
            {
                'cluster_id': t.cluster_id,
                'size': len(t.messages),
                'coherence': t.coherence_score,
                'keywords': t.keywords
            }
            for t in topics
        ]
    })


# Export Endpoints

@app.route('/api/export/json', methods=['POST'])
def export_json():
    """Export to JSON"""
    data = request.get_json()
    messages = data.get('messages', [])

    json_str = ConversationExporter.to_json(messages, pretty=True)

    return jsonify({'data': json_str})


@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Export to CSV"""
    data = request.get_json()
    messages = data.get('messages', [])

    csv_str = ConversationExporter.to_csv(messages)

    return csv_str, 200, {'Content-Type': 'text/csv'}


@app.route('/api/export/html', methods=['POST'])
def export_html():
    """Export to HTML"""
    data = request.get_json()
    messages = data.get('messages', [])
    style = data.get('style', 'default')

    html_str = ConversationExporter.to_html(messages, style=style)

    return html_str, 200, {'Content-Type': 'text/html'}


@app.route('/api/export/markdown', methods=['POST'])
def export_markdown():
    """Export to Markdown"""
    data = request.get_json()
    messages = data.get('messages', [])

    md_str = ConversationExporter.to_markdown(messages)

    return md_str, 200, {'Content-Type': 'text/markdown'}


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'systems': {
            'consciousness_engine': consciousness_engine is not None,
            'multi_agent_system': multi_agent_system is not None,
            'emotion_tracker': emotion_tracker is not None,
            'semantic_engine': semantic_engine is not None
        }
    })


def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    initialize_systems()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='AI Timestamp Context API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"Starting API server on {args.host}:{args.port}")
    run_server(args.host, args.port, args.debug)
