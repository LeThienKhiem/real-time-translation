"""
🌐 Real-time Speech Translator — Streaming Edition
====================================================
Continuously captures audio, transcribes with Whisper every ~1s using a
sliding window, and shows partial text IMMEDIATELY on screen.
Translation runs in a background thread so it never blocks the display.

Hotkeys:
    Ctrl+Shift+T  → Start/Stop
    Ctrl+Shift+M  → Switch Mic / System Audio
    Ctrl+Shift+Q  → Quit
    Ctrl+Shift+O  → Toggle overlay

Usage:
    python main.py                          # Mic → Vietnamese
    python main.py --source system          # System audio → Vietnamese
    python main.py --model small            # Better accuracy (slower)
    python main.py --model tiny             # Fastest (less accurate)
"""

import argparse
import collections
import threading
import time
import sys
import signal
import numpy as np

from audio_capture import AudioCapture
from speech_recognizer import SpeechRecognizer
from translator import Translator, LANGUAGE_NAMES
from overlay import SubtitleOverlay


class StreamingTranslator:
    """
    Streaming pipeline:
      1. Audio buffer collects audio continuously
      2. Every PROCESS_INTERVAL seconds, transcribe the last WINDOW seconds
      3. Show partial transcription on overlay IMMEDIATELY
      4. When a segment is "stable" (unchanged for 2 cycles), finalize it
      5. Translation runs in background thread → overlay updates when done
    """

    PROCESS_INTERVAL = 0.8   # Transcribe every N seconds
    WINDOW_SECONDS = 5.0     # Sliding window of audio to transcribe

    def __init__(self, source="mic", target_lang="vi", model_size="base",
                 device="cpu", overlay_position="bottom", overlay_opacity=0.85,
                 font_size=18):
        self.source = source
        self.target_lang = target_lang
        self.is_running = False
        self.is_translating = False

        print("=" * 60)
        print("🌐 Real-time Speech Translator (Streaming)")
        print("=" * 60)

        self.audio = AudioCapture(source=source, chunk_duration=0.3)
        self.recognizer = SpeechRecognizer(model_size=model_size, device=device)
        self.translator = Translator(target_language=target_lang)
        self.overlay = SubtitleOverlay(
            position=overlay_position,
            opacity=overlay_opacity,
            font_size=font_size,
        )

        # Sliding audio buffer (stores raw numpy chunks)
        self._audio_buffer = collections.deque()
        self._buffer_lock = threading.Lock()
        self._buffer_duration = 0.0  # current buffer duration in seconds

        # Translation background queue
        self._translate_queue = []  # texts waiting to be translated
        self._translate_lock = threading.Lock()

        print(f"\n📋 Configuration:")
        print(f"   Audio source:    {'Microphone' if source == 'mic' else 'System Audio'}")
        print(f"   Target language: {LANGUAGE_NAMES.get(target_lang, target_lang)}")
        print(f"   Whisper model:   {model_size}")
        print(f"   Process every:   {self.PROCESS_INTERVAL}s")
        print(f"   Audio window:    {self.WINDOW_SECONDS}s")
        print()

    # ------------------------------------------------------------------
    #  Audio collector thread — fills the sliding window buffer
    # ------------------------------------------------------------------

    def _audio_collector(self):
        """Continuously reads from audio queue into sliding window buffer."""
        chunk_dur = self.audio.chunk_duration
        max_chunks = int(self.WINDOW_SECONDS / chunk_dur) + 5  # small margin

        while self.is_translating:
            try:
                chunk = self.audio.audio_queue.get(timeout=chunk_dur * 3)
                with self._buffer_lock:
                    self._audio_buffer.append(chunk)
                    self._buffer_duration += chunk_dur
                    # Trim to keep only last WINDOW_SECONDS
                    while len(self._audio_buffer) > max_chunks:
                        self._audio_buffer.popleft()
                        self._buffer_duration = len(self._audio_buffer) * chunk_dur
            except Exception:
                continue

    def _get_buffer_audio(self):
        """Get current audio buffer as single numpy array."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            return np.concatenate(list(self._audio_buffer), axis=0)

    # ------------------------------------------------------------------
    #  Transcription thread — runs Whisper on sliding window
    # ------------------------------------------------------------------

    def _transcription_loop(self):
        """
        Continuously transcribes audio buffer every PROCESS_INTERVAL.
        Every cycle: transcribe → show on screen → translate immediately.
        No waiting for "stability" — just translate every new result.
        """
        last_translated_text = ""

        while self.is_translating:
            time.sleep(self.PROCESS_INTERVAL)

            audio_data = self._get_buffer_audio()
            if audio_data is None:
                continue

            # Check if there's actual audio (not silence)
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.003:
                continue

            # Transcribe
            result = self.recognizer.transcribe(audio_data)
            text = result["text"].strip()
            lang = result["language"]

            if not text:
                continue

            lang_name = LANGUAGE_NAMES.get(lang, lang)
            target_name = LANGUAGE_NAMES.get(self.target_lang, self.target_lang)
            self.overlay.update_lang(f"{lang_name} → {target_name}")

            # Show transcription on overlay IMMEDIATELY
            self.overlay.show_partial(f"🎙 {text}")
            print(f"\r🎙 [{lang_name}] {text[:80]:<80}", end="", flush=True)

            # Translate immediately if text is meaningfully different
            if self._is_new_text(text, last_translated_text):
                self._request_translation(text, lang)
                last_translated_text = text

    def _is_new_text(self, new_text, old_text):
        """Check if new_text is meaningfully different from old_text."""
        if not old_text:
            return True
        if new_text == old_text:
            return False
        # If the new text is significantly different (>30% changed), translate
        # This avoids re-translating minor Whisper fluctuations
        common = len(set(new_text.split()) & set(old_text.split()))
        total = max(len(new_text.split()), len(old_text.split()), 1)
        similarity = common / total
        return similarity < 0.7  # Translate if less than 70% similar

    # ------------------------------------------------------------------
    #  Translation thread — runs in background, never blocks display
    # ------------------------------------------------------------------

    def _request_translation(self, text, source_lang="auto"):
        """Queue text for background translation (replaces any pending)."""
        with self._translate_lock:
            # Only keep the LATEST request — discard old pending ones
            self._translate_queue.clear()
            self._translate_queue.append((text, source_lang))

    def _translation_loop(self):
        """Background thread that translates text as fast as possible."""
        while self.is_translating:
            item = None
            with self._translate_lock:
                if self._translate_queue:
                    item = self._translate_queue.pop(0)

            if item is None:
                time.sleep(0.05)
                continue

            text, source_lang = item
            try:
                result = self.translator.translate(text, source_language=source_lang)
                translated = result["translated"]
                print(f"\n🌐 {translated}")
                self.overlay.show_finalized(
                    original=text,
                    translated=translated,
                )
            except Exception as e:
                print(f"\n⚠️ Translation error: {e}")

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def start_translation(self):
        if self.is_translating:
            return
        self.is_translating = True
        self.audio.start()

        # Start 3 parallel threads
        threading.Thread(target=self._audio_collector, daemon=True).start()
        threading.Thread(target=self._transcription_loop, daemon=True).start()
        threading.Thread(target=self._translation_loop, daemon=True).start()

        self.overlay.update_status("LIVE", "#00ff88")
        print("▶️  Streaming started — speak now!\n")

    def stop_translation(self):
        self.is_translating = False
        self.audio.stop()
        self.overlay.update_status("PAUSED", "#888888")
        print("\n⏸️  Paused.")

    def toggle_translation(self):
        if self.is_translating:
            self.stop_translation()
        else:
            self.start_translation()

    def switch_source(self):
        was_translating = self.is_translating
        if was_translating:
            self.stop_translation()
        self.source = "system" if self.source == "mic" else "mic"
        self.audio = AudioCapture(source=self.source, chunk_duration=0.3)
        print(f"\n🔄 Switched to: {'Microphone' if self.source == 'mic' else 'System Audio'}")
        if was_translating:
            self.start_translation()

    def run(self):
        self.is_running = True
        self.overlay.start()
        time.sleep(0.5)
        self.start_translation()

        # Hotkeys
        try:
            import keyboard
            keyboard.add_hotkey("ctrl+shift+t", self.toggle_translation)
            keyboard.add_hotkey("ctrl+shift+m", self.switch_source)
            keyboard.add_hotkey("ctrl+shift+q", self.shutdown)
            keyboard.add_hotkey("ctrl+shift+o",
                lambda: self.overlay.stop() if self.overlay.is_running() else self.overlay.start())
            print("⌨️  Hotkeys: Ctrl+Shift+[T]oggle [M]ic/Sys [O]verlay [Q]uit")
        except (ImportError, Exception) as e:
            print(f"⚠️  Hotkeys unavailable: {e}")

        print("=" * 60)
        print("Press Ctrl+C to quit")
        print("=" * 60 + "\n")

        signal.signal(signal.SIGINT, lambda s, f: self.shutdown())

        try:
            while self.is_running:
                time.sleep(0.5)
                if not self.overlay.is_running():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n👋 Shutting down...")
        self.is_running = False
        self.is_translating = False
        self.audio.stop()
        self.overlay.stop()
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass
        print("✅ Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="🌐 Real-time Speech Translator (Streaming)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Mic → Vietnamese (streaming)
  python main.py --source system              # System audio → Vietnamese
  python main.py --model tiny                 # Fastest, least accurate
  python main.py --model small --device cuda  # GPU-accelerated, accurate
  python main.py --list-devices               # Show audio devices
        """
    )

    parser.add_argument("--source", choices=["mic", "system"], default="mic",
                        help="Audio source (default: mic)")
    parser.add_argument("--target", default="vi",
                        help="Target language code (default: vi)")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Compute device (default: cpu)")
    parser.add_argument("--position", default="bottom",
                        choices=["top", "bottom", "center"],
                        help="Overlay position (default: bottom)")
    parser.add_argument("--opacity", type=float, default=0.85,
                        help="Overlay opacity 0.0-1.0 (default: 0.85)")
    parser.add_argument("--font-size", type=int, default=18,
                        help="Subtitle font size (default: 18)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio devices and exit")

    args = parser.parse_args()

    if args.list_devices:
        AudioCapture.list_all_devices()
        return

    app = StreamingTranslator(
        source=args.source,
        target_lang=args.target,
        model_size=args.model,
        device=args.device,
        overlay_position=args.position,
        overlay_opacity=args.opacity,
        font_size=args.font_size,
    )
    app.run()


if __name__ == "__main__":
    main()
