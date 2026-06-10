"""
Voice guidance system using pyttsx3 (offline TTS).
Speaks alerts when bad posture or eye strain persists for more than 30 seconds.
Runs in a separate thread to avoid blocking the main monitoring loop.
"""

import time
import threading

# Try to import pyttsx3; fall back gracefully if not installed
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Messages to speak for different alert types
VOICE_MESSAGES = {
    "eye_strain": (
        "Eye strain detected. Please look away from the screen and blink a few times."
    ),
    "slouching": (
        "Poor posture detected. Please sit up straight and align your head with your shoulders."
    ),
    "break_reminder": (
        "You have been working for a while. Please take a short break to rest your eyes and stretch."
    ),
    "good_posture": (
        "Great job. Your posture looks good. Keep it up."
    ),
}

# Minimum seconds between two consecutive voice alerts (avoids repetition)
ALERT_COOLDOWN_SECONDS = 60


class VoiceGuidance:
    """
    Manages text-to-speech alerts with cooldown logic.
    Alerts fire only after a condition has persisted for TRIGGER_DELAY seconds.
    """

    TRIGGER_DELAY = 30  # seconds of continuous bad state before speaking

    def __init__(self):
        self._lock = threading.Lock()
        self._last_alert_time: dict[str, float] = {}
        self._condition_start: dict[str, float] = {}
        self._speaking = False

        # Initialize the TTS engine once
        if TTS_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 160)  # words per minute
                self._engine.setProperty("volume", 0.9)
            except Exception:
                self._engine = None
        else:
            self._engine = None

    def update_condition(self, condition_key: str, is_active: bool):
        """
        Call this every monitoring frame.
        condition_key: 'eye_strain', 'slouching', or 'break_reminder'
        is_active: True if the bad condition is currently detected.

        If the condition stays active for TRIGGER_DELAY seconds and
        the cooldown has passed, the voice alert fires.
        """
        now = time.time()

        if is_active:
            # Record when this condition first appeared
            if condition_key not in self._condition_start:
                self._condition_start[condition_key] = now

            # Check if delay threshold is reached
            elapsed = now - self._condition_start[condition_key]
            if elapsed >= self.TRIGGER_DELAY:
                self._maybe_speak(condition_key, now)
        else:
            # Reset start time when condition clears
            self._condition_start.pop(condition_key, None)

    def _maybe_speak(self, condition_key: str, now: float):
        """Speak the alert if cooldown has passed and not already speaking."""
        with self._lock:
            last = self._last_alert_time.get(condition_key, 0)
            if now - last < ALERT_COOLDOWN_SECONDS:
                return
            if self._speaking:
                return
            self._last_alert_time[condition_key] = now
            self._speaking = True

        message = VOICE_MESSAGES.get(condition_key, "Please take care of your health.")
        thread = threading.Thread(
            target=self._speak_in_thread,
            args=(message,),
            daemon=True
        )
        thread.start()

    def _speak_in_thread(self, message: str):
        """Run TTS in a background thread so the UI stays responsive."""
        try:
            if self._engine is not None:
                self._engine.say(message)
                self._engine.runAndWait()
        except Exception:
            pass
        finally:
            with self._lock:
                self._speaking = False

    def speak_now(self, condition_key: str):
        """
        Force an immediate alert, ignoring cooldown.
        Useful for manual test buttons in the UI.
        """
        message = VOICE_MESSAGES.get(condition_key, "Reminder.")
        thread = threading.Thread(
            target=self._speak_in_thread,
            args=(message,),
            daemon=True
        )
        thread.start()

    def reset_condition(self, condition_key: str):
        """Clear the start time for a condition (e.g., when session ends)."""
        self._condition_start.pop(condition_key, None)

    def reset_all(self):
        """Clear all tracked conditions and alert times."""
        with self._lock:
            self._condition_start.clear()
            self._last_alert_time.clear()

    @property
    def is_available(self) -> bool:
        """Return True if the TTS engine loaded successfully."""
        return self._engine is not None


# Singleton instance shared across the app
voice_guidance = VoiceGuidance()
