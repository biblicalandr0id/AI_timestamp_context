"""
Voice Interface Module
Provides speech-to-text and text-to-speech capabilities
Uses whisper for STT and pyttsx3/gTTS for TTS
"""

import os
import sys
import threading
from typing import Optional, Callable
from pathlib import Path
import tempfile

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("speech_recognition not available. Install with: pip install SpeechRecognition")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("pyttsx3 not available. Install with: pip install pyttsx3")

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("gtts not available. Install with: pip install gtts pygame")


class VoiceRecognizer:
    """Speech-to-text using Google Speech Recognition or Whisper"""

    def __init__(self, use_local=False):
        """
        Initialize voice recognizer

        Args:
            use_local: Use local Whisper model instead of cloud API
        """
        self.use_local = use_local
        self.recognizer = sr.Recognizer() if SPEECH_RECOGNITION_AVAILABLE else None
        self.microphone = sr.Microphone() if SPEECH_RECOGNITION_AVAILABLE else None

        # Adjust for ambient noise
        if self.microphone:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def listen_once(self, timeout=5, phrase_time_limit=10) -> Optional[str]:
        """
        Listen for a single utterance and convert to text

        Args:
            timeout: Maximum seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for the phrase

        Returns:
            Recognized text or None if failed
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("Speech recognition not available!")
            return None

        try:
            print("🎤 Listening... (speak now)")

            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            print("🔄 Recognizing...")

            # Try multiple recognition methods
            try:
                if self.use_local:
                    # Use Whisper (local)
                    text = self.recognizer.recognize_whisper(audio)
                else:
                    # Use Google Speech Recognition (cloud)
                    text = self.recognizer.recognize_google(audio)

                print(f"✅ Recognized: {text}")
                return text

            except sr.UnknownValueError:
                print("❌ Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"❌ Recognition service error: {e}")
                return None

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected within timeout")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def listen_continuous(self, callback: Callable[[str], None], stop_event: threading.Event):
        """
        Continuously listen for speech and call callback with recognized text

        Args:
            callback: Function to call with recognized text
            stop_event: Threading event to signal stop
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            print("Speech recognition not available!")
            return

        print("🎤 Continuous listening started (Ctrl+C to stop)")

        while not stop_event.is_set():
            text = self.listen_once(timeout=5, phrase_time_limit=10)
            if text:
                callback(text)


class VoiceSynthesizer:
    """Text-to-speech using pyttsx3 (offline) or gTTS (online)"""

    def __init__(self, use_online=False, voice_speed=150, voice_volume=1.0):
        """
        Initialize voice synthesizer

        Args:
            use_online: Use gTTS (online) instead of pyttsx3 (offline)
            voice_speed: Speech rate (words per minute)
            voice_volume: Volume level (0.0 to 1.0)
        """
        self.use_online = use_online
        self.voice_speed = voice_speed
        self.voice_volume = voice_volume

        # Initialize pyttsx3 engine
        if not use_online and PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', voice_speed)
            self.engine.setProperty('volume', voice_volume)

            # List available voices
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice (usually index 1)
                self.engine.setProperty('voice', voices[min(1, len(voices)-1)].id)
        else:
            self.engine = None

        # Initialize pygame for gTTS playback
        if use_online and GTTS_AVAILABLE:
            pygame.mixer.init()

    def speak(self, text: str, blocking=True) -> bool:
        """
        Convert text to speech and play

        Args:
            text: Text to speak
            blocking: Wait for speech to complete

        Returns:
            True if successful, False otherwise
        """
        if self.use_online:
            return self._speak_gtts(text, blocking)
        else:
            return self._speak_pyttsx3(text, blocking)

    def _speak_pyttsx3(self, text: str, blocking=True) -> bool:
        """Speak using pyttsx3 (offline)"""
        if not PYTTSX3_AVAILABLE or not self.engine:
            print("pyttsx3 not available!")
            return False

        try:
            self.engine.say(text)
            if blocking:
                self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False

    def _speak_gtts(self, text: str, blocking=True) -> bool:
        """Speak using gTTS (online)"""
        if not GTTS_AVAILABLE:
            print("gTTS not available!")
            return False

        try:
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                temp_path = f.name

            tts.save(temp_path)

            # Play audio
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()

            if blocking:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            # Cleanup
            os.remove(temp_path)
            return True

        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False

    def stop(self):
        """Stop current speech"""
        if self.use_online and GTTS_AVAILABLE:
            pygame.mixer.music.stop()
        elif PYTTSX3_AVAILABLE and self.engine:
            self.engine.stop()

    def set_voice_properties(self, speed: Optional[int] = None, volume: Optional[float] = None):
        """
        Change voice properties

        Args:
            speed: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        if not self.use_online and PYTTSX3_AVAILABLE and self.engine:
            if speed is not None:
                self.engine.setProperty('rate', speed)
                self.voice_speed = speed
            if volume is not None:
                self.engine.setProperty('volume', volume)
                self.voice_volume = volume


class VoiceInterface:
    """Combined voice interface with both STT and TTS"""

    def __init__(self, use_online_tts=False, use_local_stt=False):
        """
        Initialize voice interface

        Args:
            use_online_tts: Use gTTS instead of pyttsx3
            use_local_stt: Use local Whisper instead of Google API
        """
        self.recognizer = VoiceRecognizer(use_local=use_local_stt) if SPEECH_RECOGNITION_AVAILABLE else None
        self.synthesizer = VoiceSynthesizer(use_online=use_online_tts)
        self.is_listening = False
        self.listen_thread = None
        self.stop_event = threading.Event()

    def listen(self, timeout=5, phrase_time_limit=10) -> Optional[str]:
        """Listen for speech once"""
        if self.recognizer:
            return self.recognizer.listen_once(timeout, phrase_time_limit)
        return None

    def speak(self, text: str, blocking=True) -> bool:
        """Speak text"""
        return self.synthesizer.speak(text, blocking)

    def start_continuous_listening(self, callback: Callable[[str], None]):
        """Start continuous listening in background thread"""
        if self.is_listening:
            print("Already listening!")
            return

        self.stop_event.clear()
        self.is_listening = True

        self.listen_thread = threading.Thread(
            target=self.recognizer.listen_continuous,
            args=(callback, self.stop_event),
            daemon=True
        )
        self.listen_thread.start()

    def stop_continuous_listening(self):
        """Stop continuous listening"""
        if not self.is_listening:
            return

        self.stop_event.set()
        self.is_listening = False

        if self.listen_thread:
            self.listen_thread.join(timeout=2)

    def voice_conversation(self, chatbot_function: Callable[[str], str], max_turns=10):
        """
        Have a voice conversation with the chatbot

        Args:
            chatbot_function: Function that takes user input and returns bot response
            max_turns: Maximum number of conversation turns
        """
        print("🎙️ Voice Conversation Mode")
        print("=" * 50)

        self.speak("Hello! I'm ready to chat. What would you like to talk about?")

        for turn in range(max_turns):
            print(f"\n--- Turn {turn + 1} ---")

            # Listen for user input
            user_input = self.listen(timeout=10, phrase_time_limit=20)

            if not user_input:
                self.speak("I didn't catch that. Could you repeat?")
                continue

            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'bye', 'exit', 'quit', 'stop']):
                self.speak("Goodbye! It was nice talking to you.")
                break

            # Get chatbot response
            try:
                response = chatbot_function(user_input)
                print(f"🤖 Bot: {response}")

                # Speak response
                self.speak(response)

            except Exception as e:
                error_msg = "Sorry, I encountered an error processing your request."
                print(f"❌ Error: {e}")
                self.speak(error_msg)

        print("\n🎙️ Voice conversation ended")


def test_voice_interface():
    """Test voice interface"""
    print("Testing Voice Interface")
    print("=" * 50)

    # Test synthesizer
    print("\n1. Testing Text-to-Speech...")
    synth = VoiceSynthesizer(use_online=False)
    synth.speak("Hello! This is a test of the voice synthesizer.")

    # Test recognizer
    if SPEECH_RECOGNITION_AVAILABLE:
        print("\n2. Testing Speech-to-Text...")
        recognizer = VoiceRecognizer(use_local=False)
        text = recognizer.listen_once(timeout=5)
        if text:
            print(f"You said: {text}")
            synth.speak(f"You said: {text}")

    # Test combined interface
    print("\n3. Testing Voice Interface...")
    voice = VoiceInterface()

    def mock_chatbot(user_input: str) -> str:
        return f"You said: {user_input}. This is a test response."

    # Uncomment to test voice conversation
    # voice.voice_conversation(mock_chatbot, max_turns=3)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_voice_interface()
    else:
        print("Voice Interface Module")
        print("Usage: python voice_interface.py test")
        print("\nFeatures:")
        print("- Speech-to-text (Google API or Whisper)")
        print("- Text-to-speech (pyttsx3 or gTTS)")
        print("- Continuous listening mode")
        print("- Voice conversation mode")
