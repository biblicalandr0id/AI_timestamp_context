"""
Multi-Agent Conversation System
Supports multiple AI agents with different personalities and capabilities
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import numpy as np
from collections import defaultdict


class AgentPersonality(Enum):
    """Predefined agent personalities"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    PHILOSOPHICAL = "philosophical"
    PRACTICAL = "practical"


@dataclass
class AgentProfile:
    """Profile configuration for an AI agent"""
    agent_id: str
    name: str
    personality: AgentPersonality
    response_style: Dict[str, float] = field(default_factory=dict)
    expertise_domains: List[str] = field(default_factory=list)
    interaction_preferences: Dict[str, Any] = field(default_factory=dict)
    memory_capacity: int = 100

    def __post_init__(self):
        if not self.response_style:
            self.response_style = self._default_response_style()

    def _default_response_style(self) -> Dict[str, float]:
        """Generate default response style based on personality"""
        styles = {
            AgentPersonality.ANALYTICAL: {
                'verbosity': 0.6,
                'formality': 0.8,
                'creativity': 0.3,
                'empathy': 0.4,
                'directness': 0.9
            },
            AgentPersonality.CREATIVE: {
                'verbosity': 0.8,
                'formality': 0.3,
                'creativity': 0.95,
                'empathy': 0.6,
                'directness': 0.4
            },
            AgentPersonality.EMPATHETIC: {
                'verbosity': 0.7,
                'formality': 0.5,
                'creativity': 0.5,
                'empathy': 0.95,
                'directness': 0.5
            },
            AgentPersonality.TECHNICAL: {
                'verbosity': 0.5,
                'formality': 0.7,
                'creativity': 0.3,
                'empathy': 0.3,
                'directness': 0.95
            },
            AgentPersonality.PHILOSOPHICAL: {
                'verbosity': 0.9,
                'formality': 0.6,
                'creativity': 0.8,
                'empathy': 0.7,
                'directness': 0.3
            },
            AgentPersonality.PRACTICAL: {
                'verbosity': 0.4,
                'formality': 0.5,
                'creativity': 0.4,
                'empathy': 0.5,
                'directness': 0.9
            }
        }
        return styles.get(self.personality, {})


@dataclass
class AgentMessage:
    """Message from an agent"""
    agent_id: str
    content: str
    timestamp: datetime
    target_agent: Optional[str] = None  # For directed messages
    message_type: str = "response"  # response, question, statement, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Individual AI agent in the multi-agent system"""

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.conversation_history: List[AgentMessage] = []
        self.internal_state: Dict[str, Any] = {
            'engagement_level': 0.5,
            'confidence': 0.5,
            'interest_topics': set(),
            'conversation_partners': set()
        }

    def process_message(
        self,
        message: str,
        sender: str,
        timestamp: datetime,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Process incoming message and generate response

        Args:
            message: Message content
            sender: Sender ID
            timestamp: Message timestamp
            context: Additional context

        Returns:
            Agent's response message
        """
        # Update internal state
        self._update_state(message, sender)

        # Generate response based on personality
        response_content = self._generate_response(message, sender, context)

        # Determine message type
        message_type = self._classify_message_type(message)

        # Create response
        response = AgentMessage(
            agent_id=self.profile.agent_id,
            content=response_content,
            timestamp=timestamp,
            target_agent=sender if sender != "user" else None,
            message_type=message_type,
            metadata={
                'engagement': self.internal_state['engagement_level'],
                'confidence': self.internal_state['confidence'],
                'personality': self.profile.personality.value
            }
        )

        # Store in history
        self.conversation_history.append(response)

        return response

    def _update_state(self, message: str, sender: str):
        """Update agent's internal state based on message"""
        # Update engagement
        message_length = len(message.split())
        if message_length > 20:
            self.internal_state['engagement_level'] = min(1.0,
                self.internal_state['engagement_level'] + 0.1)
        elif message_length < 5:
            self.internal_state['engagement_level'] = max(0.1,
                self.internal_state['engagement_level'] - 0.1)

        # Track conversation partners
        self.internal_state['conversation_partners'].add(sender)

        # Extract potential topics
        words = message.lower().split()
        for word in words:
            if len(word) > 5:  # Likely topic word
                self.internal_state['interest_topics'].add(word)

    def _generate_response(
        self,
        message: str,
        sender: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response based on personality and context"""
        style = self.profile.response_style

        # Base response templates based on personality
        templates = {
            AgentPersonality.ANALYTICAL: [
                f"Analyzing the statement: '{message[:50]}...', I observe several key patterns.",
                f"Let me break down the core components of what {sender} mentioned.",
                f"From an analytical perspective, this raises interesting questions."
            ],
            AgentPersonality.CREATIVE: [
                f"What an intriguing perspective! This reminds me of...",
                f"I'm envisioning multiple creative interpretations here.",
                f"Let's explore this idea from unconventional angles."
            ],
            AgentPersonality.EMPATHETIC: [
                f"I understand where you're coming from, {sender}.",
                f"That must be important to you. Let me address your point carefully.",
                f"I appreciate you sharing this with me."
            ],
            AgentPersonality.TECHNICAL: [
                f"Technically speaking, the key points are:",
                f"Let me provide a precise analysis:",
                f"The technical aspects to consider include:"
            ],
            AgentPersonality.PHILOSOPHICAL: [
                f"This raises profound questions about the nature of...",
                f"Let us contemplate the deeper implications of {sender}'s words.",
                f"Philosophically, we might consider multiple perspectives here."
            ],
            AgentPersonality.PRACTICAL: [
                f"Practically speaking, here's what matters:",
                f"The actionable points are:",
                f"Let's focus on what we can do with this information."
            ]
        }

        template = np.random.choice(templates.get(self.profile.personality, ["Response:"]))

        # Adjust verbosity
        if style.get('verbosity', 0.5) > 0.7:
            template += " Furthermore, considering the broader context..."

        return f"[{self.profile.name}] {template}"

    def _classify_message_type(self, message: str) -> str:
        """Classify the type of message"""
        if '?' in message:
            return "question"
        elif any(word in message.lower() for word in ['think', 'believe', 'feel']):
            return "opinion"
        elif any(word in message.lower() for word in ['do', 'action', 'should', 'will']):
            return "action"
        else:
            return "statement"

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.profile.agent_id,
            'name': self.profile.name,
            'personality': self.profile.personality.value,
            'message_count': len(self.conversation_history),
            'engagement_level': self.internal_state['engagement_level'],
            'conversation_partners': len(self.internal_state['conversation_partners']),
            'interest_topics': list(self.internal_state['interest_topics'])[:10]
        }


class MultiAgentConversationSystem:
    """System managing multiple AI agents in conversation"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.conversation_log: List[AgentMessage] = []
        self.interaction_graph: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.turn_order: List[str] = []
        self.current_turn: int = 0

    def add_agent(self, profile: AgentProfile) -> Agent:
        """Add a new agent to the system"""
        agent = Agent(profile)
        self.agents[profile.agent_id] = agent
        self.turn_order.append(profile.agent_id)
        return agent

    def remove_agent(self, agent_id: str):
        """Remove an agent from the system"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.turn_order:
                self.turn_order.remove(agent_id)

    def process_user_message(
        self,
        message: str,
        timestamp: Optional[datetime] = None,
        target_agent: Optional[str] = None
    ) -> List[AgentMessage]:
        """
        Process a user message and get responses from agents

        Args:
            message: User message
            timestamp: Message timestamp
            target_agent: Specific agent to respond (None = all agents)

        Returns:
            List of agent responses
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        responses = []

        if target_agent:
            # Single agent response
            if target_agent in self.agents:
                response = self.agents[target_agent].process_message(
                    message, "user", timestamp
                )
                responses.append(response)
                self.conversation_log.append(response)
                self.interaction_graph["user"][target_agent] += 1
        else:
            # All agents respond
            for agent_id in self.turn_order:
                agent = self.agents[agent_id]
                response = agent.process_message(message, "user", timestamp)
                responses.append(response)
                self.conversation_log.append(response)
                self.interaction_graph["user"][agent_id] += 1
                timestamp += timedelta(seconds=1)  # Slight delay between agents

        return responses

    def agent_to_agent_dialogue(
        self,
        sender_id: str,
        receiver_id: str,
        message: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[AgentMessage]:
        """Facilitate dialogue between two agents"""
        if sender_id not in self.agents or receiver_id not in self.agents:
            return None

        if timestamp is None:
            timestamp = datetime.utcnow()

        response = self.agents[receiver_id].process_message(
            message, sender_id, timestamp
        )

        self.conversation_log.append(response)
        self.interaction_graph[sender_id][receiver_id] += 1

        return response

    def run_multi_agent_dialogue(
        self,
        initial_message: str,
        num_turns: int = 5,
        timestamp: Optional[datetime] = None
    ) -> List[AgentMessage]:
        """
        Run a multi-agent dialogue where agents respond in sequence

        Args:
            initial_message: Starting message
            num_turns: Number of dialogue turns
            timestamp: Starting timestamp

        Returns:
            Complete dialogue history
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        dialogue = []

        # User initiates
        current_message = initial_message
        current_sender = "user"

        for turn in range(num_turns):
            # Get next agent in turn order
            agent_id = self.turn_order[turn % len(self.turn_order)]
            agent = self.agents[agent_id]

            # Agent responds
            timestamp += timedelta(seconds=5)
            response = agent.process_message(current_message, current_sender, timestamp)

            dialogue.append(response)
            self.conversation_log.append(response)
            self.interaction_graph[current_sender][agent_id] += 1

            # Setup for next turn
            current_message = response.content
            current_sender = agent_id

        return dialogue

    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of agent interactions"""
        return {
            'total_messages': len(self.conversation_log),
            'agents': {agent_id: agent.get_stats() for agent_id, agent in self.agents.items()},
            'interaction_matrix': dict(self.interaction_graph),
            'most_active_agent': max(
                [(agent_id, len(agent.conversation_history)) for agent_id, agent in self.agents.items()],
                key=lambda x: x[1],
                default=("none", 0)
            )[0],
            'conversation_length': len(self.conversation_log)
        }

    def export_dialogue(self, filepath: str, format: str = 'json'):
        """Export dialogue to file"""
        if format == 'json':
            dialogue_data = [
                {
                    'agent_id': msg.agent_id,
                    'content': msg.content,
                    'timestamp': msg.timestamp.isoformat(),
                    'target': msg.target_agent,
                    'type': msg.message_type,
                    'metadata': msg.metadata
                }
                for msg in self.conversation_log
            ]

            with open(filepath, 'w') as f:
                json.dump(dialogue_data, f, indent=2)

        elif format == 'txt':
            with open(filepath, 'w') as f:
                for msg in self.conversation_log:
                    f.write(f"[{msg.timestamp}] {msg.agent_id}: {msg.content}\n\n")

    def create_agent_panel(self) -> List[AgentProfile]:
        """Create a diverse panel of agents with different personalities"""
        profiles = [
            AgentProfile(
                agent_id="agent_analytical",
                name="Dr. Logic",
                personality=AgentPersonality.ANALYTICAL,
                expertise_domains=["data_analysis", "logic", "patterns"]
            ),
            AgentProfile(
                agent_id="agent_creative",
                name="Nova",
                personality=AgentPersonality.CREATIVE,
                expertise_domains=["art", "storytelling", "innovation"]
            ),
            AgentProfile(
                agent_id="agent_empathetic",
                name="Aurora",
                personality=AgentPersonality.EMPATHETIC,
                expertise_domains=["psychology", "communication", "relationships"]
            ),
            AgentProfile(
                agent_id="agent_technical",
                name="Hex",
                personality=AgentPersonality.TECHNICAL,
                expertise_domains=["programming", "engineering", "systems"]
            ),
            AgentProfile(
                agent_id="agent_philosophical",
                name="Socrates",
                personality=AgentPersonality.PHILOSOPHICAL,
                expertise_domains=["philosophy", "ethics", "meaning"]
            )
        ]

        for profile in profiles:
            self.add_agent(profile)

        return profiles


def create_default_multi_agent_system() -> MultiAgentConversationSystem:
    """Create a multi-agent system with default agent panel"""
    system = MultiAgentConversationSystem()
    system.create_agent_panel()
    return system
