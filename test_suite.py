"""
Comprehensive Test Suite for AI Timestamp Context System
Tests all major components and functionality
"""

import unittest
from datetime import datetime, timedelta
import numpy as np
import json
import tempfile
import os

# Import all components to test
from conversation_processor import StandardConversation, TimestampedConversation
from conversation_system import EnhancedConversationSystem
from enhanced_processor import EnhancedConversationProcessor
from semantic_engine import EmbeddingEngine, TopicModeler, SemanticSearchEngine, AdvancedSemanticEngine
from multi_agent_system import AgentProfile, Agent, Multi AgentConversationSystem, AgentPersonality
from emotion_tracker import EmotionDetector, EmotionTracker, EmotionCategory
from export_utils import ConversationExporter, AnalyticsExporter, BatchExporter


class TestConversationProcessors(unittest.TestCase):
    """Test conversation processing components"""

    def setUp(self):
        self.timestamp = datetime.utcnow()

    def test_standard_conversation(self):
        """Test standard conversation processor"""
        conv = StandardConversation()
        result = conv.process_message("Hello, world!")

        self.assertIn("Processing single message", result)
        self.assertEqual(conv.current_context, "Hello, world!")

    def test_timestamped_conversation(self):
        """Test timestamped conversation processor"""
        conv = TimestampedConversation()

        messages = [
            ("First message", "user"),
            ("Second message", "ai"),
            ("Third message", "user")
        ]

        current_time = self.timestamp
        for content, user in messages:
            current_time += timedelta(seconds=10)
            conv.add_message(current_time, content, user)

        result = conv.process_timeline()

        self.assertIn("First message", result)
        self.assertIn("Second message", result)
        self.assertIn("Third message", result)
        self.assertEqual(len(conv.timeline), 3)

    def test_enhanced_conversation_system(self):
        """Test enhanced conversation system"""
        system = EnhancedConversationSystem()

        current_time = self.timestamp
        for i in range(5):
            current_time += timedelta(seconds=10)
            result = system.process_message(f"Message {i}", current_time, "user")

            self.assertEqual(result['context_depth'], i + 1)
            self.assertIn('timestamp', result)

        self.assertEqual(len(system.states), 5)

    def test_enhanced_processor(self):
        """Test enhanced conversation processor"""
        processor = EnhancedConversationProcessor()

        for i in range(3):
            result = processor.process_with_timestamp(
                f"Message {i}",
                datetime.utcnow().isoformat(),
                "user"
            )

            self.assertIn('response', result)
            self.assertIn('state', result)
            self.assertEqual(result['state']['metadata']['context_depth'], i + 1)


class TestSemanticEngine(unittest.TestCase):
    """Test semantic analysis components"""

    def setUp(self):
        self.embedding_engine = EmbeddingEngine(embedding_dim=384)
        self.messages = [
            "I love programming in Python",
            "Python is a great language",
            "Machine learning is fascinating",
            "Deep learning requires GPUs",
            "I enjoy cooking Italian food",
            "Pasta is my favorite dish"
        ]

    def test_embedding_generation(self):
        """Test embedding generation"""
        text = "Hello, world!"
        embedding = self.embedding_engine.encode(text)

        self.assertEqual(len(embedding), 384)
        self.assertIsInstance(embedding, np.ndarray)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        emb1 = self.embedding_engine.encode("I love Python")
        emb2 = self.embedding_engine.encode("Python is great")
        emb3 = self.embedding_engine.encode("I enjoy cooking")

        sim_similar = self.embedding_engine.cosine_similarity(emb1, emb2)
        sim_different = self.embedding_engine.cosine_similarity(emb1, emb3)

        # Similar texts should have higher similarity
        self.assertGreater(sim_similar, sim_different)

    def test_topic_extraction(self):
        """Test topic modeling"""
        topic_modeler = TopicModeler(self.embedding_engine)
        topics = topic_modeler.extract_topics(self.messages, n_topics=3)

        self.assertEqual(len(topics), 3)
        for topic in topics:
            self.assertGreater(len(topic.keywords), 0)
            self.assertGreater(topic.coherence_score, 0)

    def test_semantic_search(self):
        """Test semantic search"""
        search_engine = SemanticSearchEngine(self.embedding_engine)
        search_engine.index(self.messages)

        results = search_engine.search("programming languages", top_k=3)

        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)

        # Results should be sorted by similarity
        if len(results) > 1:
            self.assertGreaterEqual(results[0][1], results[1][1])

    def test_advanced_semantic_analysis(self):
        """Test advanced semantic engine"""
        engine = AdvancedSemanticEngine()
        analysis = engine.analyze_conversation(self.messages)

        self.assertEqual(analysis['message_count'], len(self.messages))
        self.assertIn('topics', analysis)
        self.assertIn('semantic_density', analysis)
        self.assertIn('conversation_flow', analysis)


class TestMultiAgentSystem(unittest.TestCase):
    """Test multi-agent conversation system"""

    def setUp(self):
        self.system = MultiAgentConversationSystem()

        # Add test agents
        profiles = [
            AgentProfile(
                agent_id="agent1",
                name="Alice",
                personality=AgentPersonality.ANALYTICAL
            ),
            AgentProfile(
                agent_id="agent2",
                name="Bob",
                personality=AgentPersonality.CREATIVE
            )
        ]

        for profile in profiles:
            self.system.add_agent(profile)

    def test_agent_creation(self):
        """Test agent creation"""
        self.assertEqual(len(self.system.agents), 2)
        self.assertIn("agent1", self.system.agents)
        self.assertIn("agent2", self.system.agents)

    def test_user_message_processing(self):
        """Test processing user messages"""
        responses = self.system.process_user_message("Hello, agents!")

        self.assertEqual(len(responses), 2)
        for response in responses:
            self.assertIsNotNone(response.content)
            self.assertIn(response.agent_id, ["agent1", "agent2"])

    def test_agent_to_agent_dialogue(self):
        """Test agent-to-agent dialogue"""
        response = self.system.agent_to_agent_dialogue(
            "agent1", "agent2", "What do you think?"
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.agent_id, "agent2")

    def test_multi_agent_dialogue(self):
        """Test multi-agent dialogue sequence"""
        dialogue = self.system.run_multi_agent_dialogue(
            "Let's discuss AI", num_turns=4
        )

        self.assertEqual(len(dialogue), 4)

    def test_interaction_summary(self):
        """Test interaction summary generation"""
        self.system.process_user_message("Test message")

        summary = self.system.get_interaction_summary()

        self.assertIn('total_messages', summary)
        self.assertIn('agents', summary)
        self.assertGreater(summary['total_messages'], 0)


class TestEmotionTracking(unittest.TestCase):
    """Test emotion tracking components"""

    def setUp(self):
        self.detector = EmotionDetector()
        self.tracker = EmotionTracker()

    def test_emotion_detection(self):
        """Test basic emotion detection"""
        test_cases = [
            ("I am very happy today!", EmotionCategory.JOY),
            ("This is terrible and awful", EmotionCategory.DISGUST),
            ("I am so scared and afraid", EmotionCategory.FEAR),
            ("I'm feeling sad and lonely", EmotionCategory.SADNESS),
        ]

        for text, expected_emotion in test_cases:
            state = self.detector.detect_emotions(text)
            # Primary emotion should be detected
            self.assertIsNotNone(state.primary_emotion)

    def test_valence_calculation(self):
        """Test valence calculation"""
        positive_state = self.detector.detect_emotions("I love this wonderful day!")
        negative_state = self.detector.detect_emotions("I hate this terrible situation!")

        # Positive text should have positive valence
        self.assertGreater(positive_state.valence, 0)
        # Negative text should have negative valence
        self.assertLess(negative_state.valence, 0)

    def test_emotion_tracking(self):
        """Test emotion tracking over time"""
        messages = [
            "I'm feeling great today!",
            "Things are going well.",
            "But now I'm worried.",
            "I'm quite anxious actually."
        ]

        for msg in messages:
            self.tracker.track(msg)

        summary = self.tracker.get_emotion_summary()

        self.assertEqual(summary['message_count'], len(messages))
        self.assertIn('distribution', summary)
        self.assertIn('current_mood', summary)

    def test_emotional_trajectory(self):
        """Test emotional trajectory tracking"""
        for i in range(5):
            self.tracker.track(f"Message {i}")

        trajectory = self.tracker.get_emotional_trajectory()

        self.assertIn('valence', trajectory)
        self.assertIn('arousal', trajectory)
        self.assertEqual(len(trajectory['valence']), 5)


class TestExportUtils(unittest.TestCase):
    """Test export utilities"""

    def setUp(self):
        self.messages = [
            {
                'user': 'Alice',
                'content': 'Hello, world!',
                'timestamp': datetime.utcnow()
            },
            {
                'user': 'Bob',
                'content': 'Hi there!',
                'timestamp': datetime.utcnow() + timedelta(seconds=5)
            }
        ]

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temp files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_json_export(self):
        """Test JSON export"""
        json_str = ConversationExporter.to_json(self.messages)

        self.assertIsInstance(json_str, str)
        data = json.loads(json_str)
        self.assertEqual(len(data), 2)

    def test_csv_export(self):
        """Test CSV export"""
        csv_str = ConversationExporter.to_csv(self.messages)

        self.assertIsInstance(csv_str, str)
        self.assertIn('user', csv_str)
        self.assertIn('content', csv_str)
        self.assertIn('Alice', csv_str)

    def test_markdown_export(self):
        """Test Markdown export"""
        md_str = ConversationExporter.to_markdown(self.messages)

        self.assertIn('# Conversation Export', md_str)
        self.assertIn('Alice', md_str)
        self.assertIn('Hello, world!', md_str)

    def test_html_export(self):
        """Test HTML export"""
        html_str = ConversationExporter.to_html(self.messages)

        self.assertIn('<html>', html_str)
        self.assertIn('Alice', html_str)
        self.assertIn('Hello, world!', html_str)

    def test_txt_export(self):
        """Test plain text export"""
        txt_str = ConversationExporter.to_txt(self.messages)

        self.assertIn('Alice', txt_str)
        self.assertIn('Hello, world!', txt_str)

    def test_batch_export(self):
        """Test batch export to multiple formats"""
        base_filename = os.path.join(self.temp_dir, 'test_export')

        exported = BatchExporter.export_all_formats(
            self.messages,
            base_filename,
            formats=['json', 'csv', 'txt']
        )

        self.assertEqual(len(exported), 3)
        for filepath in exported.values():
            self.assertTrue(os.path.exists(filepath))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_complete_conversation_workflow(self):
        """Test complete conversation workflow"""
        # Initialize system
        system = EnhancedConversationSystem()
        tracker = EmotionTracker()
        semantic_engine = AdvancedSemanticEngine()

        messages = [
            "I'm excited to start this project!",
            "Let's discuss the architecture.",
            "We need to consider scalability.",
            "I'm worried about performance.",
            "But I'm confident we can solve it!"
        ]

        current_time = datetime.utcnow()
        message_data = []

        # Process messages
        for msg in messages:
            current_time += timedelta(seconds=30)

            # Process through system
            conv_result = system.process_message(msg, current_time, "user")

            # Track emotions
            emotion_state = tracker.track(msg, current_time)

            message_data.append({
                'content': msg,
                'timestamp': current_time,
                'user': 'user',
                'context_depth': conv_result['context_depth'],
                'emotion': emotion_state.primary_emotion.value,
                'valence': emotion_state.valence
            })

        # Semantic analysis
        analysis = semantic_engine.analyze_conversation(messages)

        # Assertions
        self.assertEqual(len(message_data), 5)
        self.assertEqual(len(system.states), 5)
        self.assertEqual(analysis['message_count'], 5)

        # Export
        json_str = ConversationExporter.to_json(message_data)
        self.assertIsNotNone(json_str)


def run_tests(verbosity=2):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConversationProcessors))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiAgentSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestExportUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("=" * 60)
    print("AI TIMESTAMP CONTEXT - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()

    result = run_tests(verbosity=2)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
