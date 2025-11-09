"""
Advanced Emotion Tracking System
Goes beyond simple sentiment to track complex emotional states
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from collections import deque


class EmotionCategory(Enum):
    """Primary emotion categories"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"


class EmotionIntensity(Enum):
    """Emotion intensity levels"""
    NONE = 0
    MILD = 1
    MODERATE = 2
    STRONG = 3
    INTENSE = 4


@dataclass
class EmotionalState:
    """Represents a complex emotional state"""
    timestamp: datetime
    primary_emotion: EmotionCategory
    intensity: EmotionIntensity
    emotion_vector: Dict[EmotionCategory, float]
    valence: float  # -1 (negative) to +1 (positive)
    arousal: float  # 0 (calm) to 1 (excited)
    dominance: float  # 0 (submissive) to 1 (dominant)
    triggers: List[str]
    confidence: float


class EmotionDetector:
    """Detects emotions from text using lexical analysis"""

    def __init__(self):
        self.emotion_lexicons = self._build_emotion_lexicons()
        self.intensity_modifiers = self._build_intensity_modifiers()

    def _build_emotion_lexicons(self) -> Dict[EmotionCategory, List[str]]:
        """Build lexicons for each emotion category"""
        return {
            EmotionCategory.JOY: [
                'happy', 'joy', 'delight', 'pleased', 'glad', 'cheerful',
                'ecstatic', 'wonderful', 'great', 'amazing', 'fantastic',
                'love', 'excited', 'thrilled', 'blessed'
            ],
            EmotionCategory.SADNESS: [
                'sad', 'unhappy', 'depressed', 'miserable', 'sorrow',
                'grief', 'despair', 'melancholy', 'disappointed', 'heartbroken',
                'tears', 'crying', 'lonely', 'gloomy'
            ],
            EmotionCategory.ANGER: [
                'angry', 'mad', 'furious', 'irritated', 'annoyed',
                'rage', 'outraged', 'hostile', 'resentful', 'bitter',
                'frustrated', 'infuriated', 'hate'
            ],
            EmotionCategory.FEAR: [
                'afraid', 'scared', 'frightened', 'terrified', 'anxious',
                'worried', 'nervous', 'panic', 'dread', 'horror',
                'alarmed', 'threatened', 'insecure'
            ],
            EmotionCategory.SURPRISE: [
                'surprised', 'shocked', 'amazed', 'astonished', 'astounded',
                'startled', 'unexpected', 'sudden', 'wow', 'incredible'
            ],
            EmotionCategory.DISGUST: [
                'disgusted', 'revolted', 'repulsed', 'nauseated', 'sick',
                'gross', 'awful', 'terrible', 'horrible', 'nasty'
            ],
            EmotionCategory.TRUST: [
                'trust', 'confident', 'secure', 'safe', 'reliable',
                'dependable', 'faithful', 'loyal', 'honest', 'believe'
            ],
            EmotionCategory.ANTICIPATION: [
                'anticipate', 'expect', 'hope', 'eager', 'looking forward',
                'excited', 'ready', 'prepared', 'await', 'upcoming'
            ]
        }

    def _build_intensity_modifiers(self) -> Dict[str, float]:
        """Build intensity modifiers"""
        return {
            'very': 1.5,
            'extremely': 2.0,
            'incredibly': 2.0,
            'really': 1.3,
            'quite': 1.2,
            'somewhat': 0.7,
            'slightly': 0.5,
            'barely': 0.3,
            'absolutely': 2.0,
            'totally': 1.8,
            'completely': 1.8,
            'utterly': 2.0
        }

    def detect_emotions(self, text: str, timestamp: Optional[datetime] = None) -> EmotionalState:
        """
        Detect emotions in text

        Args:
            text: Input text
            timestamp: Optional timestamp

        Returns:
            EmotionalState object
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        text_lower = text.lower()
        words = text_lower.split()

        # Calculate emotion scores
        emotion_scores = {}
        triggers = []

        for emotion, lexicon in self.emotion_lexicons.items():
            score = 0.0
            for i, word in enumerate(words):
                if word in lexicon:
                    # Base score
                    word_score = 1.0

                    # Check for intensity modifiers
                    if i > 0 and words[i-1] in self.intensity_modifiers:
                        word_score *= self.intensity_modifiers[words[i-1]]

                    score += word_score
                    triggers.append(word)

            # Normalize by text length
            emotion_scores[emotion] = score / max(len(words), 1)

        # Find primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            max_score = emotion_scores[primary_emotion]
        else:
            primary_emotion = EmotionCategory.TRUST  # Neutral default
            max_score = 0.0

        # Calculate intensity
        intensity = self._calculate_intensity(max_score)

        # Calculate VAD (Valence, Arousal, Dominance)
        valence = self._calculate_valence(emotion_scores)
        arousal = self._calculate_arousal(emotion_scores)
        dominance = self._calculate_dominance(text)

        # Confidence based on clarity of emotion signals
        confidence = min(max_score * 2, 1.0)

        return EmotionalState(
            timestamp=timestamp,
            primary_emotion=primary_emotion,
            intensity=intensity,
            emotion_vector={k: v for k, v in emotion_scores.items()},
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            triggers=list(set(triggers))[:10],
            confidence=confidence
        )

    def _calculate_intensity(self, score: float) -> EmotionIntensity:
        """Convert score to intensity level"""
        if score < 0.01:
            return EmotionIntensity.NONE
        elif score < 0.05:
            return EmotionIntensity.MILD
        elif score < 0.15:
            return EmotionIntensity.MODERATE
        elif score < 0.30:
            return EmotionIntensity.STRONG
        else:
            return EmotionIntensity.INTENSE

    def _calculate_valence(self, emotion_scores: Dict[EmotionCategory, float]) -> float:
        """Calculate emotional valence (positive/negative)"""
        positive_emotions = [EmotionCategory.JOY, EmotionCategory.TRUST, EmotionCategory.SURPRISE]
        negative_emotions = [EmotionCategory.SADNESS, EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.DISGUST]

        pos_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        neg_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)

        total = pos_score + neg_score
        if total == 0:
            return 0.0

        return (pos_score - neg_score) / total

    def _calculate_arousal(self, emotion_scores: Dict[EmotionCategory, float]) -> float:
        """Calculate emotional arousal (calm/excited)"""
        high_arousal = [EmotionCategory.ANGER, EmotionCategory.FEAR, EmotionCategory.SURPRISE, EmotionCategory.JOY]
        low_arousal = [EmotionCategory.SADNESS, EmotionCategory.TRUST]

        high_score = sum(emotion_scores.get(e, 0) for e in high_arousal)
        low_score = sum(emotion_scores.get(e, 0) for e in low_arousal)

        total = high_score + low_score
        if total == 0:
            return 0.5

        return high_score / total

    def _calculate_dominance(self, text: str) -> float:
        """Calculate emotional dominance (submissive/dominant)"""
        dominant_words = ['will', 'must', 'should', 'command', 'demand', 'insist', 'certain', 'definitely']
        submissive_words = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'unsure', 'uncertain']

        text_lower = text.lower()

        dom_count = sum(1 for word in dominant_words if word in text_lower)
        sub_count = sum(1 for word in submissive_words if word in text_lower)

        if dom_count + sub_count == 0:
            return 0.5

        return dom_count / (dom_count + sub_count)


class EmotionTracker:
    """Tracks emotional evolution over time"""

    def __init__(self, window_size: int = 20):
        self.detector = EmotionDetector()
        self.emotion_history: deque = deque(maxlen=window_size)
        self.window_size = window_size

    def track(self, text: str, timestamp: Optional[datetime] = None) -> EmotionalState:
        """Track emotion for a new message"""
        state = self.detector.detect_emotions(text, timestamp)
        self.emotion_history.append(state)
        return state

    def get_emotional_trajectory(self) -> Dict[str, List[float]]:
        """Get the trajectory of emotional dimensions over time"""
        if not self.emotion_history:
            return {}

        return {
            'valence': [state.valence for state in self.emotion_history],
            'arousal': [state.arousal for state in self.emotion_history],
            'dominance': [state.dominance for state in self.emotion_history],
            'timestamps': [state.timestamp.isoformat() for state in self.emotion_history]
        }

    def get_emotion_distribution(self) -> Dict[str, float]:
        """Get distribution of emotions in conversation"""
        if not self.emotion_history:
            return {}

        emotion_counts = {}
        for state in self.emotion_history:
            emotion = state.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        total = len(self.emotion_history)
        return {emotion: count / total for emotion, count in emotion_counts.items()}

    def get_emotional_volatility(self) -> float:
        """Calculate emotional volatility (rate of change)"""
        if len(self.emotion_history) < 2:
            return 0.0

        changes = []
        for i in range(1, len(self.emotion_history)):
            prev_state = self.emotion_history[i-1]
            curr_state = self.emotion_history[i]

            # Calculate change in VAD space
            valence_change = abs(curr_state.valence - prev_state.valence)
            arousal_change = abs(curr_state.arousal - prev_state.arousal)
            dominance_change = abs(curr_state.dominance - prev_state.dominance)

            total_change = (valence_change + arousal_change + dominance_change) / 3
            changes.append(total_change)

        return float(np.mean(changes))

    def get_emotional_stability(self) -> float:
        """Calculate emotional stability (inverse of volatility)"""
        return 1.0 - self.get_emotional_volatility()

    def get_predominant_emotion(self) -> Optional[str]:
        """Get the most common emotion in recent history"""
        if not self.emotion_history:
            return None

        distribution = self.get_emotion_distribution()
        return max(distribution.items(), key=lambda x: x[1])[0] if distribution else None

    def get_current_mood(self) -> Dict[str, float]:
        """Get current overall mood based on recent messages"""
        if not self.emotion_history:
            return {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}

        recent_states = list(self.emotion_history)[-5:]  # Last 5 messages

        avg_valence = np.mean([s.valence for s in recent_states])
        avg_arousal = np.mean([s.arousal for s in recent_states])
        avg_dominance = np.mean([s.dominance for s in recent_states])

        return {
            'valence': float(avg_valence),
            'arousal': float(avg_arousal),
            'dominance': float(avg_dominance),
            'mood_label': self._mood_label(avg_valence, avg_arousal)
        }

    def _mood_label(self, valence: float, arousal: float) -> str:
        """Generate mood label from valence and arousal"""
        if valence > 0.3:
            if arousal > 0.6:
                return "excited/joyful"
            else:
                return "content/peaceful"
        elif valence < -0.3:
            if arousal > 0.6:
                return "stressed/angry"
            else:
                return "sad/depressed"
        else:
            if arousal > 0.6:
                return "alert/anxious"
            else:
                return "calm/neutral"

    def detect_emotional_shift(self, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        """Detect significant emotional shifts"""
        if len(self.emotion_history) < 2:
            return None

        latest = self.emotion_history[-1]
        previous = self.emotion_history[-2]

        # Calculate shift magnitude
        valence_shift = latest.valence - previous.valence
        arousal_shift = latest.arousal - previous.arousal

        shift_magnitude = np.sqrt(valence_shift**2 + arousal_shift**2)

        if shift_magnitude > threshold:
            return {
                'detected': True,
                'magnitude': float(shift_magnitude),
                'from_emotion': previous.primary_emotion.value,
                'to_emotion': latest.primary_emotion.value,
                'valence_change': float(valence_shift),
                'arousal_change': float(arousal_shift),
                'timestamp': latest.timestamp.isoformat()
            }

        return None

    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get comprehensive emotion summary"""
        return {
            'message_count': len(self.emotion_history),
            'distribution': self.get_emotion_distribution(),
            'current_mood': self.get_current_mood(),
            'volatility': self.get_emotional_volatility(),
            'stability': self.get_emotional_stability(),
            'predominant_emotion': self.get_predominant_emotion(),
            'trajectory': self.get_emotional_trajectory()
        }
