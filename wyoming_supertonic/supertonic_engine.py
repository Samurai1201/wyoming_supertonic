import inspect
import logging
from typing import Tuple

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Attempt to import language-specific text normalizers.
# If dependencies (e.g., num2words) are missing, the server will not crash,
# but will proceed without text normalization.
try:
    from .ru_norm import RussianTextNormalizer
    RU_NORMALIZER_AVAILABLE = True
except ImportError as e:
    _LOGGER.warning("Russian text normalizer won't be available: %s", e)
    RU_NORMALIZER_AVAILABLE = False


class SupertonicEngine:
    """
    Main TTS engine class that interacts with the official Supertone V3 library.

    Supports multi-language text-to-speech, custom voice styles, text
    normalization, and silence cropping to improve streaming continuity.
    """

    def __init__(
        self,
        steps: int = 5,
        speed: float = 1.0,
        model_path: str = None,
        crop_silence_ms: int = 350,
    ) -> None:
        """
        Initialize the Supertonic TTS engine configuration.

        Args:
            steps (int): Number of denoising steps for synthesis.
            speed (float): Speech speed multiplier.
            model_path (str, optional): Custom path to store models.
            crop_silence_ms (int): Amount of silence to crop from both ends.
        """
        self.steps = steps
        self.speed = speed
        self.model_path = model_path
        self.crop_silence_ms = crop_silence_ms
        self.tts = None
        self.sample_rate = 44100  # V3 defaults to 44.1kHz output

        # Official V3 voices
        self.available_voices =[
            "M1", "M2", "M3", "M4", "M5",
            "F1", "F2", "F3", "F4", "F5"
        ]

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

    def load(self) -> None:
        """
        Load the TTS models via the official V3 library.

        Models and voice presets are automatically downloaded on the first run
        if they are not already present in the Hugging Face cache.

        Raises:
            RuntimeError: If the 'supertonic' package is not installed.
        """
        _LOGGER.info("Loading engine (Supertonic V3)...")

        try:
            from supertonic import TTS
            _LOGGER.info(
                "Initializing TTS (automatic model download is enabled)..."
            )
            # The library handles model storage in the HF cache directory
            self.tts = TTS(auto_download=True)
            _LOGGER.info(
                "Engine ready. Sample Rate: %dHz. Voices: %d",
                self.sample_rate,
                len(self.available_voices)
            )

        except ImportError as exc:
            raise RuntimeError(
                "supertonic package not found! Please install it via pip. "
                f"Error: {exc}"
            ) from exc

    def synthesize(
        self, text: str, voice_name: str, lang_code: str = "en"
    ) -> Tuple[bytes, int]:
        """
        Synthesize text into PCM audio bytes.

        Args:
            text (str): The input string to speak (may contain expression tags).
            voice_name (str): The ID of the voice preset (e.g., 'M1').
            lang_code (str): Two-letter ISO language code.

        Returns:
            Tuple[bytes, int]: A tuple containing raw PCM int16 bytes and
            the audio sample rate.

        Raises:
            RuntimeError: If the engine has not been loaded yet.
            Exception: If synthesis fails inside the library.
        """
        if self.tts is None:
            raise RuntimeError("Engine not loaded! Call load() first.")

        # 1. Check and process language code
        if not lang_code:
            lang_code = "en"
        short_lang = lang_code[:2].lower()
        if short_lang not in self.supported_langs:
            short_lang = "en"

        # 2. Text Normalization (Numbers, symbols, etc.)
        # Note: English words inside Russian text are preserved for the model.
        if short_lang in self.normalizers:
            original_text = text
            text = self.normalizers[short_lang].normalize(text)
            if original_text != text:
                _LOGGER.debug(
                    "[%s] Normalized: '%s' -> '%s'",
                    short_lang,
                    original_text,
                    text
                )

        # 3. Resolve Voice Style Preset
        try:
            style = self.tts.get_voice_style(voice_name=voice_name)
        except Exception as exc:
            _LOGGER.warning(
                "Voice '%s' not found. Falling back to 'M1'. (%s)",
                voice_name,
                exc
            )
            style = self.tts.get_voice_style(voice_name="M1")

        # 4. Perform Synthesis
        try:
            _LOGGER.debug(
                "Synthesizing: (Voice: %s, Lang: %s, Speed: %s, Steps: %s)",
                voice_name,
                short_lang,
                self.speed,
                self.steps
            )

            kwargs = {"lang": short_lang}
            sig = inspect.signature(self.tts.synthesize)

            # Dynamically check API parameters to maintain compatibility
            if "speed" in sig.parameters:
                kwargs["speed"] = self.speed
            if "total_step" in sig.parameters:
                kwargs["total_step"] = self.steps
            elif "total_steps" in sig.parameters:
                kwargs["total_steps"] = self.steps

            wav, _ = self.tts.synthesize(text, voice_style=style, **kwargs)

            # Squeeze output to 1D array if necessary
            if hasattr(wav, "squeeze"):
                wav = wav.squeeze()

            # --- SILENCE CROPPING BLOCK ---
            if self.crop_silence_ms > 0:
                crop_samples = int(
                    (self.crop_silence_ms / 1000.0) * self.sample_rate
                )

                # Protect against over-cropping very short audio
                if len(wav) > crop_samples * 2:
                    wav = wav[crop_samples : -crop_samples]
                    _LOGGER.debug(
                        "Cropped %dms (%d samples)",
                        self.crop_silence_ms,
                        crop_samples
                    )
                else:
                    _LOGGER.warning(
                        "Audio is too short (%d samples) to crop "
                        "%d samples. Skipping crop.",
                        len(wav),
                        crop_samples * 2
                    )
            # ------------------------------

        except Exception as exc:
            _LOGGER.error("In-library synthesis failure: %s", exc)
            raise

        # 5. Convert float32[-1.0, 1.0] to int16 PCM (Wyoming standard)
        audio_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)

        return audio_int16.tobytes(), self.sample_rate