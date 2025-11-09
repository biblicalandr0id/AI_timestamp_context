"""
Web-based Chatbot Server with Continual Learning
Flask server with WebSocket support for real-time chat
"""

from flask import Flask, request, jsonify, render_template_string, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Optional
import logging
import json

from rag_system import create_rag_system, RAGConfig
from neural_chatbot import ChatbotConfig
from knowledge_store import KnowledgeGraphManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global RAG system instance
rag_system = None
active_sessions: Dict[str, Dict] = {}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot - Continual Learning</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }

        .chat-header .status {
            font-size: 12px;
            opacity: 0.9;
        }

        .chat-stats {
            background: #f8f9fa;
            padding: 10px 20px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.bot {
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message.bot .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .message-meta {
            font-size: 11px;
            margin-top: 5px;
            opacity: 0.7;
        }

        .bot .message-meta {
            display: flex;
            gap: 10px;
            color: #666;
        }

        .confidence-badge {
            background: #4caf50;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 10px;
        }

        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .chat-input-wrapper {
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: #667eea;
        }

        .send-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: transform 0.2s;
        }

        .send-button:hover {
            transform: scale(1.05);
        }

        .send-button:active {
            transform: scale(0.95);
        }

        .feedback-buttons {
            display: flex;
            gap: 5px;
            margin-top: 8px;
        }

        .feedback-btn {
            background: #f0f0f0;
            border: none;
            padding: 4px 10px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }

        .feedback-btn:hover {
            background: #e0e0e0;
        }

        .feedback-btn.positive:hover {
            background: #4caf50;
            color: white;
        }

        .feedback-btn.negative:hover {
            background: #f44336;
            color: white;
        }

        .typing-indicator {
            display: none;
            padding: 10px 18px;
            background: white;
            border-radius: 18px;
            width: 60px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .typing-indicator.active {
            display: block;
        }

        .typing-indicator span {
            height: 8px;
            width: 8px;
            background: #999;
            border-radius: 50%;
            display: inline-block;
            margin-right: 3px;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🧠 AI Chatbot</h1>
            <div class="status">Powered by Neural Networks & Continual Learning</div>
        </div>

        <div class="chat-stats">
            <span id="stats">Conversations: 0 | Knowledge Items: 0 | Confidence: 0%</span>
        </div>

        <div class="chat-messages" id="messages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm an AI chatbot that learns from every conversation. Ask me anything!
                </div>
            </div>
        </div>

        <div class="chat-input-container">
            <div class="chat-input-wrapper">
                <input type="text" class="chat-input" id="messageInput"
                       placeholder="Type your message..." autofocus>
                <button class="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let sessionId = localStorage.getItem('sessionId') || generateUUID();
        localStorage.setItem('sessionId', sessionId);

        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        socket.on('connect', () => {
            console.log('Connected to server');
            socket.emit('join', {session_id: sessionId});
        });

        socket.on('response', (data) => {
            removeTypingIndicator();
            addMessage('bot', data.response, {
                confidence: data.confidence,
                sources: data.retrieved_knowledge_count,
                messageId: data.message_id
            });
            updateStats(data.stats);
        });

        socket.on('stats_update', (data) => {
            updateStats(data);
        });

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (message) {
                addMessage('user', message);
                addTypingIndicator();

                socket.emit('message', {
                    message: message,
                    session_id: sessionId
                });

                input.value = '';
            }
        }

        function addMessage(type, content, meta = {}) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;

            let metaHTML = '';
            if (type === 'bot' && meta.confidence) {
                const confPercent = (meta.confidence * 100).toFixed(0);
                metaHTML = `
                    <div class="message-meta">
                        <span class="confidence-badge">${confPercent}% confident</span>
                        <span>${meta.sources || 0} sources</span>
                    </div>
                    <div class="feedback-buttons">
                        <button class="feedback-btn positive" onclick="sendFeedback('${meta.messageId}', 1)">👍 Helpful</button>
                        <button class="feedback-btn negative" onclick="sendFeedback('${meta.messageId}', 0)">👎 Not helpful</button>
                    </div>
                `;
            }

            messageDiv.innerHTML = `
                <div class="message-content">
                    ${content}
                    ${metaHTML}
                </div>
            `;

            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function addTypingIndicator() {
            const messagesDiv = document.getElementById('messages');
            const indicator = document.createElement('div');
            indicator.className = 'message bot';
            indicator.id = 'typing-indicator';
            indicator.innerHTML = `
                <div class="typing-indicator active">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            `;
            messagesDiv.appendChild(indicator);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function removeTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            if (indicator) {
                indicator.remove();
            }
        }

        function sendFeedback(messageId, rating) {
            socket.emit('feedback', {
                message_id: messageId,
                rating: rating,
                session_id: sessionId
            });
        }

        function updateStats(stats) {
            const statsSpan = document.getElementById('stats');
            statsSpan.textContent = `Conversations: ${stats.conversations || 0} | ` +
                                  `Knowledge Items: ${stats.knowledge_items || 0} | ` +
                                  `Confidence: ${(stats.avg_confidence * 100).toFixed(0)}%`;
        }

        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Request initial stats
        socket.emit('get_stats', {session_id: sessionId});
    </script>
</body>
</html>
"""


def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system

    logger.info("Initializing RAG system...")

    chatbot_config = ChatbotConfig(
        model_name="microsoft/DialoGPT-small",  # Start with small model
        learning_rate=5e-5,
        batch_size=4,
        experience_replay_size=500
    )

    rag_config = RAGConfig(
        retrieval_top_k=5,
        knowledge_weight=0.7,
        min_relevance_score=0.3
    )

    rag_system = create_rag_system(
        db_path="chatbot_knowledge.db",
        chatbot_config=chatbot_config,
        rag_config=rag_config
    )

    logger.info("RAG system initialized successfully")


def start_continual_learning_thread():
    """Start background thread for continual learning"""
    def continual_learning_loop():
        while True:
            time.sleep(300)  # Every 5 minutes

            if rag_system:
                try:
                    logger.info("Running continual learning cycle...")
                    rag_system.batch_learn_from_conversations(batch_size=10)
                    logger.info("Continual learning cycle completed")
                except Exception as e:
                    logger.error(f"Error in continual learning: {e}")

    thread = threading.Thread(target=continual_learning_loop, daemon=True)
    thread.start()
    logger.info("Continual learning thread started")


@app.route('/')
def index():
    """Serve the chat interface"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_system_loaded': rag_system is not None,
        'active_sessions': len(active_sessions),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 503

    stats = rag_system.get_comprehensive_stats()
    return jsonify(stats)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('join')
def handle_join(data):
    """Handle session join"""
    session_id = data.get('session_id', str(uuid.uuid4()))

    if session_id not in active_sessions:
        active_sessions[session_id] = {
            'conversation_history': [],
            'created_at': datetime.utcnow(),
            'message_count': 0
        }

    logger.info(f"Session joined: {session_id}")

    # Send initial stats
    if rag_system:
        stats = rag_system.chatbot.get_stats()
        emit('stats_update', {
            'conversations': stats['total_conversations'],
            'knowledge_items': stats['knowledge_items'],
            'avg_confidence': stats.get('average_confidence', 0.5)
        })


@socketio.on('message')
def handle_message(data):
    """Handle incoming chat message"""
    if not rag_system:
        emit('response', {
            'response': 'System is initializing, please wait...',
            'confidence': 0.0,
            'message_id': str(uuid.uuid4())
        })
        return

    message = data.get('message', '')
    session_id = data.get('session_id', 'default')

    if not message:
        return

    # Get session
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            'conversation_history': [],
            'created_at': datetime.utcnow(),
            'message_count': 0
        }

    session_data = active_sessions[session_id]

    try:
        # Generate response using RAG
        result = rag_system.generate_with_retrieval(
            user_input=message,
            conversation_history=session_data['conversation_history'],
            user_id=session_id
        )

        # Update conversation history
        session_data['conversation_history'].append({
            'role': 'user',
            'content': message
        })
        session_data['conversation_history'].append({
            'role': 'bot',
            'content': result['response']
        })
        session_data['message_count'] += 1

        # Generate message ID for feedback
        message_id = str(uuid.uuid4())

        # Store message ID for feedback
        if 'messages' not in session_data:
            session_data['messages'] = {}
        session_data['messages'][message_id] = {
            'user_input': message,
            'bot_response': result['response'],
            'timestamp': datetime.utcnow()
        }

        # Send response
        emit('response', {
            'response': result['response'],
            'confidence': result['confidence'],
            'retrieved_knowledge_count': result['retrieved_knowledge_count'],
            'message_id': message_id,
            'stats': {
                'conversations': rag_system.chatbot.stats['total_conversations'],
                'knowledge_items': rag_system.chatbot.stats['knowledge_items'],
                'avg_confidence': result['confidence']
            }
        })

    except Exception as e:
        logger.error(f"Error handling message: {e}")
        emit('response', {
            'response': f'Sorry, I encountered an error: {str(e)}',
            'confidence': 0.0,
            'message_id': str(uuid.uuid4())
        })


@socketio.on('feedback')
def handle_feedback(data):
    """Handle user feedback on responses"""
    message_id = data.get('message_id')
    rating = data.get('rating', 0.5)
    session_id = data.get('session_id', 'default')

    if session_id not in active_sessions:
        return

    session_data = active_sessions[session_id]
    if 'messages' not in session_data or message_id not in session_data['messages']:
        return

    message_data = session_data['messages'][message_id]

    # Learn from feedback
    try:
        rag_system.chatbot.learn_from_interaction(
            user_input=message_data['user_input'],
            bot_response=message_data['bot_response'],
            rating=float(rating)
        )

        logger.info(f"Learned from feedback: rating={rating}")

        emit('feedback_received', {'status': 'success'})

    except Exception as e:
        logger.error(f"Error handling feedback: {e}")


@socketio.on('get_stats')
def handle_get_stats(data):
    """Send current stats to client"""
    if rag_system:
        stats = rag_system.chatbot.get_stats()
        emit('stats_update', {
            'conversations': stats['total_conversations'],
            'knowledge_items': stats['knowledge_items'],
            'avg_confidence': stats.get('average_confidence', 0.5)
        })


def run_chatbot_server(host='0.0.0.0', port=5000, debug=False):
    """Run the chatbot server"""
    print("=" * 60)
    print("🧠 NEURAL CHATBOT SERVER")
    print("=" * 60)
    print(f"\n→ Initializing RAG system...")

    initialize_rag_system()

    print(f"✓ RAG system initialized")
    print(f"→ Starting continual learning thread...")

    start_continual_learning_thread()

    print(f"✓ Continual learning thread started")
    print(f"\n🚀 Server starting on http://{host}:{port}")
    print(f"→ Open your browser to start chatting!")
    print("=" * 60 + "\n")

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Neural Chatbot Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    run_chatbot_server(args.host, args.port, args.debug)
