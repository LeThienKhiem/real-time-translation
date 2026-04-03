"""
Speech Recognition Module
Uses faster-whisper for efficient speech-to-text conversion.
Automatically detects the spoken language.
"""

import numpy as np
import time


class SpeechRecognizer:
    """Speech-to-text using faster-whisper with auto language detection."""

    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Args:
            model_size: Whisper model size - "tiny", "base", "small", "medium", "large-v3"
                        Larger = more accurate but slower
                        Recommended: "base" for real-time, "small" for better accuracy
            device: "cpu" or "cuda" (for NVIDIA GPU)
            compute_type: "int8" (fast), "float16" (GPU), "float32" (accurate)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            from faster_whisper import WhisperModel
            print(f"📦 Loading Whisper model '{self.model_size}' on {self.device}...")
            start = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed = time.time() - start
            print(f"✅ Model loaded in {elapsed:.1f}s")
        except ImportError:
            print("❌ faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def transcribe(self, audio_data, language=None):
        """
        Transcribe audio data to text.

        Args:
            audio_data: numpy array of audio (float32, 16kHz)
            language: Force a specific language (None = auto-detect)

        Returns:
            dict with keys: text, language, confidence, duration
        """
        if self.model is None:
            return {"text": "", "language": "unknown", "confidence": 0.0, "duration": 0.0}

        # Ensure correct format
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        audio_data = audio_data.astype(np.float32)

        # Amplify quiet audio (e.g. AirPods mic) to a good level for Whisper
        max_val = np.max(np.abs(audio_data))
        rms_before = float(np.sqrt(np.mean(audio_data ** 2)))

        if max_val > 1e-6:
            # Normalize to peak = 0.9
            audio_data = audio_data * (0.9 / max_val)

        rms_after = float(np.sqrt(np.mean(audio_data ** 2)))
        print(f"      [Whisper] pre-amp RMS={rms_before:.5f} peak={max_val:.5f} → post-amp RMS={rms_after:.4f}")

        start = time.time()

        try:
            # NO VAD filter — process ALL audio, let Whisper decide
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=5,
                vad_filter=False,
            )

            # Collect all segment texts
            texts = []
            for segment in segments:
                texts.append(segment.text.strip())

            full_text = " ".join(texts)
            elapsed = time.time() - start

            return {
                "text": full_text,
                "language": info.language if info.language else "unknown",
                "confidence": info.language_probability if info.language_probability else 0.0,
                "duration": elapsed,
            }

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return {"text": "", "language": "unknown", "confidence": 0.0, "duration": 0.0}

    def get_supported_languages(self):
        """Return list of languages Whisper can detect."""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
        ]
