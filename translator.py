"""
Translation Module
Uses deep-translator for translating text between languages.
"""

from deep_translator import GoogleTranslator
import time


# Whisper language code → Google Translate language code
# Whisper uses short codes, Google Translate sometimes needs different ones
WHISPER_TO_GOOGLE = {
    "zh": "zh-CN",      # Chinese → Simplified Chinese
    "jw": "jv",          # Javanese
    "nn": "no",          # Norwegian Nynorsk → Norwegian
    "he": "iw",          # Hebrew (Whisper uses "he", Google uses "iw")
}


def map_lang_code(whisper_code):
    """Convert Whisper language code to Google Translate compatible code."""
    return WHISPER_TO_GOOGLE.get(whisper_code, whisper_code)


# Language code mapping for display
LANGUAGE_NAMES = {
    "en": "English", "vi": "Tiếng Việt", "zh": "中文", "zh-CN": "中文简体",
    "zh-TW": "中文繁體", "ja": "日本語",
    "ko": "한국어", "fr": "Français", "de": "Deutsch", "es": "Español",
    "pt": "Português", "ru": "Русский", "ar": "العربية", "hi": "हिन्दी",
    "th": "ไทย", "id": "Bahasa Indonesia", "ms": "Bahasa Melayu",
    "it": "Italiano", "nl": "Nederlands", "pl": "Polski", "tr": "Türkçe",
    "sv": "Svenska", "da": "Dansk", "fi": "Suomi", "no": "Norsk",
    "uk": "Українська", "cs": "Čeština", "ro": "Română", "hu": "Magyar",
    "el": "Ελληνικά", "he": "עברית", "bn": "বাংলা", "ta": "தமிழ்",
    "te": "తెలుగు", "ml": "മലയാളം", "ka": "ქართული",
}


class Translator:
    """Real-time text translator using Google Translate."""

    def __init__(self, target_language="vi"):
        """
        Args:
            target_language: Target language code (default: "vi" for Vietnamese)
        """
        self.target_language = target_language
        self._cache = {}  # Simple translation cache
        self._max_cache_size = 500

    def translate(self, text, source_language="auto"):
        """
        Translate text to target language.

        Args:
            text: Text to translate
            source_language: Source language code ("auto" for auto-detect)

        Returns:
            dict with keys: original, translated, source_lang, target_lang, duration
        """
        if not text or not text.strip():
            return {
                "original": "",
                "translated": "",
                "source_lang": source_language,
                "target_lang": self.target_language,
                "duration": 0.0,
            }

        text = text.strip()

        # Map Whisper language codes to Google Translate codes
        source_language = map_lang_code(source_language)
        target = map_lang_code(self.target_language)

        # Skip translation if source == target
        if source_language == target:
            return {
                "original": text,
                "translated": text,
                "source_lang": source_language,
                "target_lang": target,
                "duration": 0.0,
                "skipped": True,
            }

        # Check cache
        cache_key = f"{source_language}:{target}:{text}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached["from_cache"] = True
            return cached

        start = time.time()

        try:
            translator = GoogleTranslator(
                source=source_language,
                target=target,
            )
            translated = translator.translate(text)
            elapsed = time.time() - start

            result = {
                "original": text,
                "translated": translated if translated else text,
                "source_lang": source_language,
                "target_lang": target,
                "duration": elapsed,
            }

            # Cache the result
            if len(self._cache) >= self._max_cache_size:
                # Remove oldest entries
                keys = list(self._cache.keys())
                for k in keys[:100]:
                    del self._cache[k]
            self._cache[cache_key] = result

            return result

        except Exception as e:
            elapsed = time.time() - start
            print(f"⚠️  Translation error: {e}")
            return {
                "original": text,
                "translated": f"[Translation error: {e}]",
                "source_lang": source_language,
                "target_lang": target,
                "duration": elapsed,
                "error": str(e),
            }

    def set_target_language(self, lang_code):
        """Change the target language."""
        self.target_language = lang_code
        self._cache.clear()  # Clear cache when language changes
        print(f"🌐 Target language changed to: {LANGUAGE_NAMES.get(lang_code, lang_code)}")

    @staticmethod
    def get_language_name(code):
        """Get display name for a language code."""
        return LANGUAGE_NAMES.get(code, code)

    @staticmethod
    def get_supported_targets():
        """Get list of supported target languages."""
        try:
            langs = GoogleTranslator.get_supported_languages(as_dict=True)
            return langs
        except Exception:
            return LANGUAGE_NAMES
