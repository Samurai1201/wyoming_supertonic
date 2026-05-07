import logging
import numpy as np
import os
import inspect

_LOGGER = logging.getLogger(__name__)

# Attempt to import language-specific text normalizers.
# If dependencies (e.g., num2words) are missing, the server will not crash,
# but will proceed without text normalization.
try:
    from .ru_norm import RussianTextNormalizer
    RU_NORMALIZER_AVAILABLE = True
except ImportError as e:
    _LOGGER.warning(f"Russian text normalizer won't be available: {e}")
    RU_NORMALIZER_AVAILABLE = False


class SupertonicEngine:
    def __init__(self, steps: int = 5, speed: float = 1.0, model_path: str = None):
        self.steps = steps
        self.speed = speed
        self.model_path = model_path
        self.tts = None
        self.sample_rate = 44100  # V3 defaults to 44.1kHz output
        
        # Official V3 voices
        self.available_voices =["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
        
        # Full list of supported V3 languages
        self.supported_langs =[
            "en", "ko", "ja", "ar", "bg", "cs", "da", "de", 
            "el", "es", "et", "fi", "fr", "hi", "hr", "hu", 
            "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", 
            "ru", "sk", "sl", "sv", "tr", "uk", "vi"
        ]

        # Normalizer registry by language (extensible architecture)
        self.normalizers = {}
        
        if RU_NORMALIZER_AVAILABLE:
            self.normalizers["ru"] = RussianTextNormalizer()
            # Future normalizers can be added here, e.g.:
            # self.normalizers["es"] = SpanishTextNormalizer()

    def load(self):
        """Load via the official V3 library"""
        _LOGGER.info("Loading engine (Supertonic V3)...")
        
        try:
            from supertonic import TTS
            _LOGGER.info("Initializing TTS (Models and presets will auto-download if missing)...")
            self.tts = TTS(auto_download=True)
            _LOGGER.info(f"Engine ready. Rate: {self.sample_rate}Hz. Voices: {len(self.available_voices)}")
            
        except ImportError as e:
            raise RuntimeError(f"supertonic package not found! Please run 'pip install supertonic'. Error: {e}")

    def synthesize(self, text: str, voice_name: str, lang_code: str = "en") -> tuple[bytes, int]:
        if self.tts is None:
            raise RuntimeError("Engine not loaded!")

        # 1. Check and process language code
        if not lang_code: 
            lang_code = "en"
        short_lang = lang_code[:2].lower()
        if short_lang not in self.supported_langs:
            short_lang = "en"

        # 2. Text normalization for specific language (if registered)
        if short_lang in self.normalizers:
            original_text = text
            text = self.normalizers[short_lang].normalize(text)
            if original_text != text:
                _LOGGER.debug(f"[{short_lang}] Normalized text: '{original_text}' -> '{text}'")

        # 3. Get voice style (preset)
        try:
            style = self.tts.get_voice_style(voice_name=voice_name)
        except Exception as e:
            _LOGGER.warning(f"Voice style '{voice_name}' not found, defaulting to M1. ({e})")
            style = self.tts.get_voice_style(voice_name="M1")

        # 4. Synthesize via library
        try:
            _LOGGER.debug(f"Synthesizing: (Voice: {voice_name}, Lang: {short_lang}, Speed: {self.speed}, Steps: {self.steps})")
            
            kwargs = {"lang": short_lang}
            sig = inspect.signature(self.tts.synthesize)
            
            # Safely pass arguments depending on the library's exact API
            if "speed" in sig.parameters:
                kwargs["speed"] = self.speed
            if "total_step" in sig.parameters:
                kwargs["total_step"] = self.steps
            elif "total_steps" in sig.parameters:
                kwargs["total_steps"] = self.steps
                
            wav, duration = self.tts.synthesize(text, voice_style=style, **kwargs)
            
            # Remove batch dimensions, e.g., (1, N) -> (N,)
            if hasattr(wav, "squeeze"):
                wav = wav.squeeze()
            
        except Exception as e:
            _LOGGER.error(f"Synthesis error in library: {e}")
            raise e

        # 5. Convert float32[-1.0, 1.0] array to int16 PCM format for Wyoming
        audio_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
        
        return audio_int16.tobytes(), self.sample_rate