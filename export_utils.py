"""
Export Utilities for Various Formats
Supports JSON, CSV, Markdown, HTML, and more
"""

import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import StringIO
import base64


class ConversationExporter:
    """Export conversations to various formats"""

    @staticmethod
    def to_json(
        messages: List[Dict[str, Any]],
        filepath: Optional[str] = None,
        pretty: bool = True
    ) -> str:
        """
        Export to JSON format

        Args:
            messages: List of message dictionaries
            filepath: Optional file path to save
            pretty: Whether to pretty-print JSON

        Returns:
            JSON string
        """
        # Convert datetime objects to ISO format
        serializable_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            for key, value in msg_copy.items():
                if isinstance(value, datetime):
                    msg_copy[key] = value.isoformat()
            serializable_messages.append(msg_copy)

        indent = 2 if pretty else None
        json_str = json.dumps(serializable_messages, indent=indent)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    @staticmethod
    def to_csv(
        messages: List[Dict[str, Any]],
        filepath: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> str:
        """
        Export to CSV format

        Args:
            messages: List of message dictionaries
            filepath: Optional file path to save
            columns: Specific columns to export

        Returns:
            CSV string
        """
        if not messages:
            return ""

        # Determine columns
        if columns is None:
            columns = list(messages[0].keys())

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')

        writer.writeheader()
        for msg in messages:
            # Convert datetime objects
            row = {}
            for col in columns:
                value = msg.get(col)
                if isinstance(value, datetime):
                    row[col] = value.isoformat()
                elif isinstance(value, (list, dict)):
                    row[col] = json.dumps(value)
                else:
                    row[col] = value
            writer.writerow(row)

        csv_str = output.getvalue()
        output.close()

        if filepath:
            with open(filepath, 'w') as f:
                f.write(csv_str)

        return csv_str

    @staticmethod
    def to_markdown(
        messages: List[Dict[str, Any]],
        filepath: Optional[str] = None,
        include_metadata: bool = False
    ) -> str:
        """
        Export to Markdown format

        Args:
            messages: List of message dictionaries
            filepath: Optional file path to save
            include_metadata: Whether to include metadata

        Returns:
            Markdown string
        """
        lines = []

        lines.append("# Conversation Export")
        lines.append(f"\nExported: {datetime.utcnow().isoformat()}")
        lines.append(f"\nTotal Messages: {len(messages)}")
        lines.append("\n---\n")

        for i, msg in enumerate(messages, 1):
            # Header
            user = msg.get('user', msg.get('agent_id', 'Unknown'))
            timestamp = msg.get('timestamp', '')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            lines.append(f"## Message {i}")
            lines.append(f"\n**From:** {user}")
            lines.append(f"**Time:** {timestamp}\n")

            # Content
            content = msg.get('content', '')
            lines.append(f"{content}\n")

            # Metadata
            if include_metadata:
                lines.append("\n**Metadata:**")
                for key, value in msg.items():
                    if key not in ['content', 'user', 'agent_id', 'timestamp']:
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, indent=2)
                        lines.append(f"- {key}: {value}")

            lines.append("\n---\n")

        markdown = "\n".join(lines)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(markdown)

        return markdown

    @staticmethod
    def to_html(
        messages: List[Dict[str, Any]],
        filepath: Optional[str] = None,
        title: str = "Conversation Export",
        style: str = "default"
    ) -> str:
        """
        Export to HTML format

        Args:
            messages: List of message dictionaries
            filepath: Optional file path to save
            title: HTML page title
            style: Style theme ('default', 'dark', 'minimal')

        Returns:
            HTML string
        """
        styles = {
            'default': """
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
                .message { background: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { display: flex; justify-content: space-between; margin-bottom: 10px; color: #666; font-size: 0.9em; }
                .user { font-weight: bold; color: #2196F3; }
                .timestamp { color: #999; }
                .content { line-height: 1.6; }
                h1 { color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }
            """,
            'dark': """
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #1e1e1e; color: #e0e0e0; }
                .message { background: #2d2d2d; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 3px solid #4CAF50; }
                .header { display: flex; justify-content: space-between; margin-bottom: 10px; font-size: 0.9em; }
                .user { font-weight: bold; color: #4CAF50; }
                .timestamp { color: #888; }
                .content { line-height: 1.6; }
                h1 { color: #4CAF50; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            """,
            'minimal': """
                body { font-family: 'Courier New', monospace; max-width: 800px; margin: 0 auto; padding: 20px; }
                .message { border-left: 2px solid #000; padding-left: 15px; margin: 20px 0; }
                .header { margin-bottom: 5px; }
                .user { font-weight: bold; }
                .timestamp { font-size: 0.85em; color: #666; }
                .content { margin-top: 10px; }
            """
        }

        html_parts = []
        html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>{styles.get(style, styles['default'])}</style>
</head>
<body>
    <h1>{title}</h1>
    <p>Total Messages: {len(messages)} | Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}</p>
""")

        for msg in messages:
            user = msg.get('user', msg.get('agent_id', 'Unknown'))
            timestamp = msg.get('timestamp', '')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            content = msg.get('content', '')

            html_parts.append(f"""
    <div class="message">
        <div class="header">
            <span class="user">{user}</span>
            <span class="timestamp">{timestamp}</span>
        </div>
        <div class="content">{content}</div>
    </div>
""")

        html_parts.append("""
</body>
</html>
""")

        html = "".join(html_parts)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(html)

        return html

    @staticmethod
    def to_txt(
        messages: List[Dict[str, Any]],
        filepath: Optional[str] = None,
        format: str = "standard"
    ) -> str:
        """
        Export to plain text format

        Args:
            messages: List of message dictionaries
            filepath: Optional file path to save
            format: Format style ('standard', 'chat', 'transcript')

        Returns:
            Text string
        """
        lines = []

        if format == "standard":
            lines.append("=" * 60)
            lines.append("CONVERSATION EXPORT")
            lines.append(f"Exported: {datetime.utcnow().isoformat()}")
            lines.append(f"Total Messages: {len(messages)}")
            lines.append("=" * 60)
            lines.append("")

            for msg in messages:
                user = msg.get('user', msg.get('agent_id', 'Unknown'))
                timestamp = msg.get('timestamp', '')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

                content = msg.get('content', '')

                lines.append(f"[{timestamp}] {user}")
                lines.append(content)
                lines.append("-" * 60)
                lines.append("")

        elif format == "chat":
            for msg in messages:
                user = msg.get('user', msg.get('agent_id', 'Unknown'))
                content = msg.get('content', '')
                lines.append(f"{user}: {content}")
                lines.append("")

        elif format == "transcript":
            for i, msg in enumerate(messages, 1):
                user = msg.get('user', msg.get('agent_id', 'Unknown'))
                timestamp = msg.get('timestamp', '')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.strftime('%H:%M:%S')

                content = msg.get('content', '')
                lines.append(f"{i:03d} [{timestamp}] {user}")
                lines.append(f"    {content}")
                lines.append("")

        text = "\n".join(lines)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(text)

        return text


class AnalyticsExporter:
    """Export analytics and metrics"""

    @staticmethod
    def export_consciousness_metrics(
        states: List[Any],
        filepath: str,
        format: str = 'json'
    ):
        """Export consciousness metrics"""
        metrics = []

        for state in states:
            if hasattr(state, 'to_dict'):
                metrics.append(state.to_dict())
            elif isinstance(state, dict):
                metrics.append(state)

        if format == 'json':
            ConversationExporter.to_json(metrics, filepath)
        elif format == 'csv':
            ConversationExporter.to_csv(metrics, filepath)

    @staticmethod
    def export_emotion_data(
        emotion_tracker: Any,
        filepath: str,
        format: str = 'json'
    ):
        """Export emotion tracking data"""
        summary = emotion_tracker.get_emotion_summary()

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format == 'csv':
            # Flatten for CSV
            flat_data = []
            trajectory = summary.get('trajectory', {})
            valence = trajectory.get('valence', [])
            arousal = trajectory.get('arousal', [])
            timestamps = trajectory.get('timestamps', [])

            for i in range(len(valence)):
                flat_data.append({
                    'timestamp': timestamps[i] if i < len(timestamps) else '',
                    'valence': valence[i],
                    'arousal': arousal[i] if i < len(arousal) else 0,
                    'dominance': trajectory.get('dominance', [0])[i] if i < len(trajectory.get('dominance', [])) else 0
                })

            ConversationExporter.to_csv(flat_data, filepath)

    @staticmethod
    def export_multi_agent_stats(
        multi_agent_system: Any,
        filepath: str,
        format: str = 'json'
    ):
        """Export multi-agent system statistics"""
        stats = multi_agent_system.get_interaction_summary()

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)


class BatchExporter:
    """Batch export to multiple formats"""

    @staticmethod
    def export_all_formats(
        messages: List[Dict[str, Any]],
        base_filename: str,
        formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Export to multiple formats at once

        Args:
            messages: List of message dictionaries
            base_filename: Base filename without extension
            formats: List of formats to export (default: all)

        Returns:
            Dictionary mapping format to filepath
        """
        if formats is None:
            formats = ['json', 'csv', 'markdown', 'html', 'txt']

        exported = {}

        for fmt in formats:
            filepath = f"{base_filename}.{fmt}"

            if fmt == 'json':
                ConversationExporter.to_json(messages, filepath)
            elif fmt == 'csv':
                ConversationExporter.to_csv(messages, filepath)
            elif fmt == 'markdown' or fmt == 'md':
                filepath = f"{base_filename}.md"
                ConversationExporter.to_markdown(messages, filepath)
            elif fmt == 'html':
                ConversationExporter.to_html(messages, filepath)
            elif fmt == 'txt':
                ConversationExporter.to_txt(messages, filepath)

            exported[fmt] = filepath

        return exported
