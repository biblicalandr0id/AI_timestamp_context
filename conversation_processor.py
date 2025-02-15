from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    timestamp: datetime
    content: str
    user: str

class StandardConversation:
    def __init__(self):
        self.current_context = ""
    
    def process_message(self, message: str) -> str:
        # Simulates "pulling paper from hat" - limited context
        self.current_context = message
        return f"Processing single message: {message}"

class TimestampedConversation:
    def __init__(self):
        self.timeline: Dict[datetime, Message] = {}
    
    def add_message(self, timestamp: datetime, content: str, user: str):
        self.timeline[timestamp] = Message(timestamp, content, user)
    
    def process_timeline(self) -> str:
        # Processes entire conversation timeline at once
        sorted_messages = sorted(self.timeline.items())
        complete_context = "\n".join(
            f"[{msg.timestamp}] {msg.user}: {msg.content}"
            for _, msg in sorted_messages
        )
        return f"Processing complete timeline:\n{complete_context}"