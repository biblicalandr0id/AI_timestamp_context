#!/usr/bin/env python3
"""
Basic test script without heavy dependencies
"""

from datetime import datetime, timedelta
import sys

# Test conversation system (no heavy dependencies)
try:
    from conversation_system import EnhancedConversationSystem, ConversationState
    from conversation_processor import StandardConversation, TimestampedConversation
    from enhanced_processor import EnhancedConversationProcessor

    print("✓ Basic imports successful")

    # Test 1: Standard Conversation
    print("\nTest 1: Standard Conversation")
    print("-" * 50)
    standard = StandardConversation()
    result = standard.process_message("Hello")
    print(f"Result: {result}")

    # Test 2: Timestamped Conversation
    print("\nTest 2: Timestamped Conversation")
    print("-" * 50)
    timestamped = TimestampedConversation()
    current_time = datetime.utcnow()

    messages = [
        "First message",
        "Second message",
        "Third message"
    ]

    for msg in messages:
        current_time += timedelta(seconds=10)
        timestamped.add_message(current_time, msg, "user")

    result = timestamped.process_timeline()
    print(result)

    # Test 3: Enhanced Conversation System
    print("\nTest 3: Enhanced Conversation System")
    print("-" * 50)
    system = EnhancedConversationSystem()

    current_time = datetime.utcnow()
    for i, msg in enumerate(["Hello", "How are you?", "Testing timestamps"]):
        current_time += timedelta(seconds=15)
        result = system.process_message(msg, current_time, "user")
        print(f"Message {i+1}: Depth={result['context_depth']}")

    # Test 4: Enhanced Processor
    print("\nTest 4: Enhanced Processor")
    print("-" * 50)
    processor = EnhancedConversationProcessor()

    for i in range(3):
        result = processor.process_with_timestamp(
            f"Message {i+1}",
            datetime.utcnow().isoformat(),
            "user"
        )
        print(f"Context depth: {result['state']['metadata']['context_depth']}")

    print("\n" + "=" * 50)
    print("✓ All basic tests passed!")
    print("=" * 50)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
