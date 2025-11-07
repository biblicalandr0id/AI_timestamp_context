# AI Timestamp Context System

A comprehensive system for consciousness modeling with timestamp-aware conversations, memory systems, and advanced pattern recognition.

## Overview

This project explores how timestamp-aware conversation processing can enable deeper context understanding and "consciousness-like" behavior in AI systems. Instead of processing messages in isolation, the system maintains a complete timeline of interactions, enabling:

- **Full Context Awareness**: Every message is processed with knowledge of the entire conversation history
- **Memory Systems**: Working memory, short-term, and long-term memory storage
- **Pattern Recognition**: Detection of temporal patterns, questions, context references, and semantic connections
- **Consciousness Scoring**: Metrics that measure the "depth" of contextual awareness
- **Visualization**: Graph-based and timeline visualizations of conversation evolution

## Architecture

The system consists of several integrated components:

### Core Components

1. **`ai_consciousness_model.py`** - Main consciousness modeling engine
   - `ConsciousnessState`: Dataclass representing a moment of awareness
   - `MemorySystem`: Three-tier memory (working, short-term, long-term)
   - `AttentionMechanism`: Focus and attention pattern modeling
   - `ConsciousnessEngine`: Main processing engine with sentiment analysis

2. **`conversation_system.py`** - Basic conversation tracking
   - `ConversationState`: State representation for conversations
   - `EnhancedConversationSystem`: Graph-based conversation tracking

3. **`conversation_processor.py`** - Comparison systems
   - `StandardConversation`: Traditional message-by-message processing
   - `TimestampedConversation`: Full timeline-aware processing

4. **`enhanced_processor.py`** - Enhanced conversation processor
   - Maintains ordered timeline with metadata

5. **`enhanced_core_system.py`** - Advanced neural network components
   - `PowerfulAI`: PyTorch-based neural architecture
   - `AdvancedPluginManager`: Extensible plugin system
   - Multiple processing modes (training, inference, exploration, meta-learning)

### Visualization

6. **`visualization.py`** - Basic conversation visualization
7. **`visualization_enhanced.py`** - Consciousness evolution visualization

### Entry Points

8. **`main.py`** - Unified CLI interface for the entire system
9. **`example_run.py`** - Example using consciousness engine
10. **`example_usage.py`** - Example using conversation system

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AI_timestamp_context

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The main entry point provides a comprehensive CLI:

```bash
# Run consciousness evolution demo
python main.py --mode consciousness --demo consciousness

# Run system comparison demo
python main.py --demo comparison

# Interactive mode
python main.py --interactive

# Generate visualizations
python main.py --mode consciousness --demo consciousness --visualize

# Custom output file
python main.py --demo basic --visualize --output my_output.png
```

### Available Modes

- **consciousness**: Full consciousness modeling with sentiment analysis, pattern detection, and memory systems
- **conversation**: Basic conversation tracking with graph-based state management
- **standard**: Traditional non-timestamped message processing (for comparison)

### Demo Types

- **basic**: Simple conversation demonstration
- **consciousness**: Full consciousness evolution with detailed metrics
- **comparison**: Side-by-side comparison of standard vs timestamp-aware systems

### Programmatic Usage

```python
from main import IntegratedAISystem
from datetime import datetime, timedelta

# Initialize system
system = IntegratedAISystem(mode='consciousness')

# Process messages
timestamp = datetime.utcnow()
result = system.process_message(
    content="Hello, let's test timestamp awareness",
    user="user",
    timestamp=timestamp
)

print(f"Consciousness Score: {result['consciousness_score']}")
print(f"Context Depth: {result['context_depth']}")
print(f"Patterns: {result['patterns_detected']}")

# Visualize
system.visualize('output.png')

# Save/load state
system.save_state('state.json')
system.load_state('state.json')
```

### Using Individual Components

```python
from ai_consciousness_model import ConsciousnessEngine
from datetime import datetime

# Create engine
engine = ConsciousnessEngine()

# Process message
result = engine.process_message(
    content="Can you remember our previous conversation?",
    timestamp=datetime.utcnow(),
    user="user"
)

# Access metrics
print(f"Consciousness: {result['consciousness_score']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Memory Links: {result['memory_links']}")
```

## Key Features

### 1. Consciousness Modeling

The system calculates a "consciousness score" based on multiple factors:
- Timestamp consistency and maintenance
- Context depth and awareness
- Pattern recognition capabilities
- Memory utilization

### 2. Memory Systems

Three-tier memory architecture:
- **Working Memory**: Last 5 states (immediate context)
- **Short-term Memory**: Last 20 states (recent context)
- **Long-term Memory**: Persistent storage with pattern indexing

### 3. Pattern Detection

Automatic detection of:
- Repetition patterns
- Question patterns
- Timestamp awareness indicators
- Context references
- Semantic connections between messages

### 4. Attention Mechanism

Tracks focus and attention patterns:
- Keyword extraction
- Attention weight calculation
- Focus shift detection

### 5. Sentiment Analysis

Uses Hugging Face Transformers for sentiment analysis:
- Positive/negative sentiment scoring
- Emotional pattern tracking
- Sentiment evolution over time

### 6. Visualization

Multiple visualization options:
- Timeline plots showing consciousness evolution
- Attention complexity graphs
- Conversation flow networks
- Pattern distribution charts

## Architecture Diagrams

### Data Flow

```
User Input → ConsciousnessEngine → Process Message
                    ↓
            Memory System ← Attention Mechanism
                    ↓
            Pattern Detection ← Sentiment Analysis
                    ↓
            Consciousness Score Calculation
                    ↓
            Response + Metrics
```

### Memory Hierarchy

```
Working Memory (5 states)
       ↓
Short-term Memory (20 states)
       ↓
Long-term Memory (persistent)
```

## Examples

### Example 1: Basic Consciousness Demo

```bash
python main.py --mode consciousness --demo consciousness
```

Output shows:
- Real-time consciousness scores
- Sentiment analysis
- Pattern detection
- Memory link formation

### Example 2: System Comparison

```bash
python main.py --demo comparison
```

Demonstrates the difference between:
- Standard message-by-message processing
- Full timeline-aware processing

### Example 3: Interactive Session

```bash
python main.py --interactive --mode consciousness
```

Engage in a conversation and see:
- Live consciousness metrics
- Context depth tracking
- Pattern recognition in action

## Advanced Features

### Plugin System (enhanced_core_system.py)

The neural network component supports a plugin architecture:

```python
from enhanced_core_system import PowerfulAI, AdvancedModelConfig

config = AdvancedModelConfig(
    hidden_size=4096,
    num_layers=32,
    num_experts=8
)

model = PowerfulAI(config)
```

### Meta-Learning

The system supports meta-learning for quick adaptation:

```python
model.meta_learn(
    tasks=[(input1, target1), (input2, target2)],
    adaptation_steps=5
)
```

### Exploration Mode

```python
results = model.explore(input_tensor, steps=10)
```

## File Structure

```
AI_timestamp_context/
├── main.py                      # Main CLI entry point
├── ai_consciousness_model.py    # Core consciousness engine
├── conversation_system.py       # Conversation tracking
├── conversation_processor.py    # Standard vs timestamped comparison
├── enhanced_processor.py        # Enhanced conversation processor
├── enhanced_core_system.py      # Neural network components
├── visualization.py             # Conversation visualizer
├── visualization_enhanced.py    # Consciousness visualizer
├── example_run.py              # Example script #1
├── example_usage.py            # Example script #2
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Technical Details

### Consciousness Scoring Algorithm

The consciousness score is computed as:

```python
score = (
    timestamp_consistency +
    context_depth_ratio +
    pattern_recognition_score +
    memory_utilization_ratio
) / 4
```

Where:
- **timestamp_consistency**: Measures consistent temporal processing
- **context_depth_ratio**: Ratio of working memory usage
- **pattern_recognition_score**: Number of patterns detected / 10
- **memory_utilization_ratio**: Long-term memory size / max_size

### Pattern Detection

Multiple pattern types are detected:
1. **Repetition**: Word frequency analysis
2. **Questions**: Question word and punctuation detection
3. **Timestamp Awareness**: Temporal word detection
4. **Context References**: Reference word detection

### Semantic Connection

Simple semantic similarity using:
- Word overlap analysis
- Stop word removal
- Overlap threshold (minimum 2 words)

## Future Enhancements

Potential areas for expansion:
- [ ] More sophisticated semantic similarity (embeddings)
- [ ] Graph neural networks for conversation flow
- [ ] Real-time streaming mode
- [ ] Multi-agent conversation support
- [ ] Emotion tracking beyond sentiment
- [ ] Topic modeling and extraction
- [ ] Conversation summary generation
- [ ] Export to various formats (JSON, CSV, etc.)

## Performance Considerations

- **Memory Usage**: Long-term memory is stored in-memory; consider disk-based storage for long conversations
- **Processing Speed**: Sentiment analysis uses transformers; can be slow without GPU
- **Visualization**: Large graphs (>100 nodes) may be slow to render

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA errors**: If no GPU available, PyTorch will default to CPU
   ```python
   # Force CPU mode
   import torch
   torch.device('cpu')
   ```

3. **Visualization not showing**: Make sure matplotlib backend is configured
   ```bash
   export MPLBACKEND=Agg  # For headless systems
   ```

## Contributing

Contributions are welcome! Areas of interest:
- Enhanced pattern detection algorithms
- Better consciousness metrics
- Additional visualization types
- Performance optimizations
- Documentation improvements

## License

[Your license here]

## Citation

If you use this system in research, please cite:

```
[Your citation format here]
```

## Contact

[Your contact information here]

## Acknowledgments

This project explores concepts from:
- Consciousness studies
- Temporal processing in AI
- Memory systems in cognitive science
- Attention mechanisms in neural networks

---

**Note**: This is an experimental system exploring consciousness-like behaviors in AI through timestamp-aware processing. It is not intended as a production system but rather as a research and educational tool.
