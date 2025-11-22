import os
from pathlib import Path
from typing import Optional

from manim import logger

from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import SpeechService

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None
    logger.error(
        "Missing 'google-cloud-texttospeech' package. "
        "Run `pip install google-cloud-texttospeech` to use GoogleTextToSpeechService."
    )


class GoogleTextToSpeechService(SpeechService):
    """
    Speech service using Google Cloud Text-to-Speech API.
    Supports standard WaveNet/Neural2 voices and the new Gemini-based generative voices.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        lang: str = "en-US",
        voice_name: str = "Charon",
        model_name: str = "gemini-2.5-pro-tts",
        location: str = "global",
        style_prompt: Optional[str] = None,
        credentials_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            project_id (str, optional): Google Cloud Project ID. Defaults to environment variable.
            lang (str, optional): Language code (e.g., "en-US"). Defaults to "en-US".
            voice_name (str, optional): Name of the voice (e.g., "Charon", "en-US-Neural2-A").
            model_name (str, optional): Model name (e.g., "gemini-2.5-pro-tts").
            location (str, optional): API location. Defaults to "global".
            style_prompt (str, optional): Default style prompt for generative models (e.g., "Speak consistently").
            credentials_path (str, optional): Path to service account JSON key.
            **kwargs: Additional arguments passed to SpeechService.
        """
        kwargs["transcription_model"] = kwargs.get("transcription_model", "base")

        super().__init__(**kwargs)

        self.lang = lang
        self.voice_name = voice_name
        self.model_name = model_name
        self.location = location
        self.style_prompt = style_prompt
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self.client = None

    def _ensure_client(self):
        """Lazy initialization of the Google TTS client."""
        if self.client is None:
            if texttospeech is None:
                raise ImportError(
                    "Google Cloud Text-to-Speech library is not installed. "
                    "Please run `pip install google-cloud-texttospeech`."
                )
            try:
                self.client = texttospeech.TextToSpeechClient()
            except Exception as e:
                logger.error(f"Failed to initialize Google TTS Client: {e}")
                raise

    def generate_from_text(
        self, text: str, cache_dir: str | None = None, path: str | None = None, **kwargs
    ) -> dict:
        """
        Generates audio from text using Google Cloud TTS.

        Args:
            text (str): The text to synthesize.
            cache_dir (str, optional): Override for cache directory.
            path (str, optional): Specific output path.
            **kwargs: Can include 'prompt', 'voice_name', 'model_name', 'lang'.
        """
        self._ensure_client()

        prompt = kwargs.get("prompt", self.style_prompt)
        lang = kwargs.get("lang", self.lang)
        voice_name = kwargs.get("voice_name", self.voice_name)
        model_name = kwargs.get("model_name", self.model_name)

        text = remove_bookmarks(text)
        input_data = {
            "input_text": text,
            "service": "google_tts",
            "model": model_name,
            "voice": voice_name,
            "lang": lang,
            "prompt": prompt,
        }

        base_cache_dir = cache_dir or self.cache_dir

        cached_result = self.get_cached_result(input_data, base_cache_dir)
        if cached_result is not None:
            logger.info(f"Using cached voiceover for: '{text[:30]}...'")
            return cached_result

        if path is None:
            path = self.get_audio_basename(input_data) + ".mp3"

        try:
            if prompt:
                synthesis_input = texttospeech.SynthesisInput(text=text, prompt=prompt)
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)

            voice_params = texttospeech.VoiceSelectionParams(
                language_code=lang, name=voice_name, model_name=model_name
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )

            logger.info(
                f"Synthesizing with Google TTS ({model_name} / {voice_name})..."
            )
            response = self.client.synthesize_speech(
                input=synthesis_input, voice=voice_params, audio_config=audio_config
            )

            with (Path(base_cache_dir) / path).open("wb") as out:
                out.write(response.audio_content)
                logger.info(f"Saved voiceover to: {path}")

        except Exception as e:
            logger.error(f"Google TTS generation failed: {e}")
            raise

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": path,
        }

        return json_dict
