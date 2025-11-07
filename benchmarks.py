"""
Performance Benchmarking Suite
Measures and analyzes system performance across all components
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Import components to benchmark
from conversation_system import EnhancedConversationSystem
from semantic_engine import AdvancedSemanticEngine
from multi_agent_system import create_default_multi_agent_system
from emotion_tracker import EmotionTracker


@dataclass
class BenchmarkResult:
    """Benchmark result data"""
    test_name: str
    duration_seconds: float
    operations_count: int
    ops_per_second: float
    memory_mb: float
    cpu_percent: float
    metadata: Dict[str, Any]


class PerformanceBenchmark:
    """Performance benchmarking tools"""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()

    def measure(
        self,
        name: str,
        func: Callable,
        args=None,
        kwargs=None,
        iterations: int = 1
    ) -> BenchmarkResult:
        """
        Measure performance of a function

        Args:
            name: Benchmark name
            func: Function to benchmark
            args: Positional arguments
            kwargs: Keyword arguments
            iterations: Number of iterations

        Returns:
            BenchmarkResult
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        # Warm up
        func(*args, **kwargs)

        # Measure
        self.process.cpu_percent()  # Reset CPU measurement
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        for _ in range(iterations):
            func(*args, **kwargs)

        end_time = time.time()

        mem_after = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self.process.cpu_percent()

        duration = end_time - start_time
        ops_per_second = iterations / duration if duration > 0 else 0

        result = BenchmarkResult(
            test_name=name,
            duration_seconds=duration,
            operations_count=iterations,
            ops_per_second=ops_per_second,
            memory_mb=mem_after - mem_before,
            cpu_percent=cpu_percent,
            metadata={}
        )

        self.results.append(result)
        return result

    def benchmark_conversation_system(self, message_count: int = 100):
        """Benchmark conversation system"""
        system = EnhancedConversationSystem()

        def process_messages():
            current_time = datetime.utcnow()
            for i in range(message_count):
                current_time += timedelta(seconds=1)
                system.process_message(f"Test message {i}", current_time, "user")

        result = self.measure(
            f"ConversationSystem_{message_count}_messages",
            process_messages
        )

        result.metadata['message_count'] = message_count
        result.metadata['messages_per_second'] = message_count / result.duration_seconds

        return result

    def benchmark_semantic_engine(self, message_count: int = 50):
        """Benchmark semantic analysis engine"""
        engine = AdvancedSemanticEngine()
        messages = [f"This is test message number {i}" for i in range(message_count)]

        def analyze():
            engine.analyze_conversation(messages)

        result = self.measure(
            f"SemanticEngine_{message_count}_messages",
            analyze
        )

        result.metadata['message_count'] = message_count
        result.metadata['analysis_time'] = result.duration_seconds

        return result

    def benchmark_embedding_generation(self, text_count: int = 100):
        """Benchmark embedding generation"""
        engine = AdvancedSemanticEngine()
        texts = [f"Sample text number {i} for embedding generation" for i in range(text_count)]

        def generate_embeddings():
            for text in texts:
                engine.embedding_engine.encode(text, use_cache=False)

        result = self.measure(
            f"EmbeddingGeneration_{text_count}_texts",
            generate_embeddings
        )

        result.metadata['text_count'] = text_count
        result.metadata['embeddings_per_second'] = text_count / result.duration_seconds

        return result

    def benchmark_multi_agent_system(self, dialogue_turns: int = 20):
        """Benchmark multi-agent system"""
        system = create_default_multi_agent_system()

        def run_dialogue():
            system.run_multi_agent_dialogue("Let's discuss AI", num_turns=dialogue_turns)

        result = self.measure(
            f"MultiAgent_{dialogue_turns}_turns",
            run_dialogue
        )

        result.metadata['dialogue_turns'] = dialogue_turns
        result.metadata['turns_per_second'] = dialogue_turns / result.duration_seconds

        return result

    def benchmark_emotion_tracking(self, message_count: int = 100):
        """Benchmark emotion tracking"""
        tracker = EmotionTracker()
        messages = [
            "I'm feeling great today!",
            "This is concerning.",
            "I'm excited about the future!",
            "That's disappointing.",
            "What an amazing opportunity!"
        ] * (message_count // 5)

        def track_emotions():
            for msg in messages:
                tracker.track(msg)

        result = self.measure(
            f"EmotionTracking_{message_count}_messages",
            track_emotions
        )

        result.metadata['message_count'] = message_count
        result.metadata['analyses_per_second'] = message_count / result.duration_seconds

        return result

    def benchmark_topic_extraction(self, message_counts: List[int] = [10, 50, 100]):
        """Benchmark topic extraction with different message counts"""
        results = []

        for count in message_counts:
            engine = AdvancedSemanticEngine()
            messages = [f"Topic test message {i} about various subjects" for i in range(count)]

            def extract_topics():
                engine.topic_modeler.extract_topics(messages, n_topics=min(5, count // 2))

            result = self.measure(
                f"TopicExtraction_{count}_messages",
                extract_topics
            )

            result.metadata['message_count'] = count
            results.append(result)

        return results

    def benchmark_semantic_search(self, corpus_size: int = 100, queries: int = 10):
        """Benchmark semantic search"""
        engine = AdvancedSemanticEngine()
        corpus = [f"Document {i} with various content about topic {i % 10}" for i in range(corpus_size)]

        engine.search_engine.index(corpus)

        queries_list = [f"Query about topic {i}" for i in range(queries)]

        def search():
            for query in queries_list:
                engine.search_engine.search(query, top_k=5)

        result = self.measure(
            f"SemanticSearch_{corpus_size}docs_{queries}queries",
            search
        )

        result.metadata['corpus_size'] = corpus_size
        result.metadata['queries'] = queries
        result.metadata['searches_per_second'] = queries / result.duration_seconds

        return result

    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE PERFORMANCE BENCHMARK")
        print("=" * 60 + "\n")

        # Conversation system benchmarks
        print("→ Benchmarking Conversation System...")
        for count in [10, 50, 100]:
            result = self.benchmark_conversation_system(count)
            print(f"  {count} messages: {result.ops_per_second:.2f} ops/sec, "
                  f"{result.duration_seconds:.3f}s")

        # Semantic engine benchmarks
        print("\n→ Benchmarking Semantic Engine...")
        for count in [10, 25, 50]:
            result = self.benchmark_semantic_engine(count)
            print(f"  {count} messages: {result.duration_seconds:.3f}s")

        # Embedding generation
        print("\n→ Benchmarking Embedding Generation...")
        for count in [50, 100]:
            result = self.benchmark_embedding_generation(count)
            print(f"  {count} embeddings: {result.metadata['embeddings_per_second']:.2f} emb/sec")

        # Multi-agent system
        print("\n→ Benchmarking Multi-Agent System...")
        for turns in [10, 20]:
            result = self.benchmark_multi_agent_system(turns)
            print(f"  {turns} turns: {result.duration_seconds:.3f}s")

        # Emotion tracking
        print("\n→ Benchmarking Emotion Tracking...")
        for count in [50, 100]:
            result = self.benchmark_emotion_tracking(count)
            print(f"  {count} analyses: {result.metadata['analyses_per_second']:.2f} analyses/sec")

        # Topic extraction
        print("\n→ Benchmarking Topic Extraction...")
        results = self.benchmark_topic_extraction([10, 50, 100])
        for result in results:
            print(f"  {result.metadata['message_count']} messages: {result.duration_seconds:.3f}s")

        # Semantic search
        print("\n→ Benchmarking Semantic Search...")
        result = self.benchmark_semantic_search(100, 10)
        print(f"  100 docs, 10 queries: {result.metadata['searches_per_second']:.2f} searches/sec")

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {'message': 'No benchmark results available'}

        total_duration = sum(r.duration_seconds for r in self.results)
        avg_memory = np.mean([r.memory_mb for r in self.results])
        avg_cpu = np.mean([r.cpu_percent for r in self.results])

        return {
            'summary': {
                'total_tests': len(self.results),
                'total_duration_seconds': total_duration,
                'average_memory_mb': avg_memory,
                'average_cpu_percent': avg_cpu
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'duration_seconds': r.duration_seconds,
                    'ops_per_second': r.ops_per_second,
                    'memory_mb': r.memory_mb,
                    'cpu_percent': r.cpu_percent,
                    'metadata': r.metadata
                }
                for r in self.results
            ],
            'fastest_test': min(self.results, key=lambda x: x.duration_seconds).test_name,
            'slowest_test': max(self.results, key=lambda x: x.duration_seconds).test_name,
            'most_memory_intensive': max(self.results, key=lambda x: x.memory_mb).test_name
        }

    def export_report(self, filepath: str):
        """Export benchmark report to JSON"""
        report = self.generate_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Benchmark report exported to: {filepath}")

    def print_summary(self):
        """Print benchmark summary"""
        report = self.generate_report()

        if 'message' in report:
            print(report['message'])
            return

        summary = report['summary']

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total Tests:      {summary['total_tests']}")
        print(f"Total Duration:   {summary['total_duration_seconds']:.2f}s")
        print(f"Avg Memory:       {summary['average_memory_mb']:.2f} MB")
        print(f"Avg CPU:          {summary['average_cpu_percent']:.1f}%")
        print(f"\nFastest Test:     {report['fastest_test']}")
        print(f"Slowest Test:     {report['slowest_test']}")
        print(f"Memory Intensive: {report['most_memory_intensive']}")
        print("=" * 60)


class MemoryProfiler:
    """Memory profiling utilities"""

    @staticmethod
    def profile_memory_growth(func: Callable, iterations: int = 100):
        """Profile memory growth over iterations"""
        process = psutil.Process()
        memory_samples = []

        for i in range(iterations):
            mem_before = process.memory_info().rss / 1024 / 1024
            func()
            mem_after = process.memory_info().rss / 1024 / 1024

            memory_samples.append({
                'iteration': i,
                'memory_mb': mem_after,
                'growth_mb': mem_after - mem_before
            })

        return memory_samples


def main():
    """Run benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(description='Performance Benchmarking Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive benchmark')
    parser.add_argument('--export', type=str, help='Export report to file')

    args = parser.parse_args()

    benchmark = PerformanceBenchmark()

    if args.comprehensive:
        benchmark.run_comprehensive_benchmark()
    elif args.quick:
        print("Running quick benchmark...")
        benchmark.benchmark_conversation_system(10)
        benchmark.benchmark_emotion_tracking(20)
    else:
        benchmark.run_comprehensive_benchmark()

    benchmark.print_summary()

    if args.export:
        benchmark.export_report(args.export)


if __name__ == '__main__':
    main()
