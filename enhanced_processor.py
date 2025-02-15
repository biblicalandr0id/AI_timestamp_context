from typing import Optional
import json

class EnhancedConversationProcessor:
    def __init__(self):
        self.conversation_state = {
            'messages': [],
            'metadata': {
                'last_timestamp': None,
                'current_user': None,
                'context_depth': 0
            }
        }
    
    def process_with_timestamp(self, message: str, timestamp: str, user: str) -> dict:
        # Maintains ordered timeline and processes with full context
        self.conversation_state['messages'].append({
            'timestamp': timestamp,
            'content': message,
            'user': user
        })
        
        self.conversation_state['metadata'].update({
            'last_timestamp': timestamp,
            'current_user': user,
            'context_depth': len(self.conversation_state['messages'])
        })
        
        return {
            'response': self._generate_response(),
            'state': self.conversation_state
        }
    
    def _generate_response(self) -> str:
        # Simulates response generation with full context awareness
        context_size = len(self.conversation_state['messages'])
        return f"Processing with {context_size} messages in context"