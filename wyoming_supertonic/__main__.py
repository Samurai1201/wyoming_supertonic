import argparse
import asyncio
import logging
import os
import sys
from functools import partial
from urllib.parse import urlparse

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncTcpServer

from . import __version__
from .supertonic_engine import SupertonicEngine
from .handler import SupertonicEventHandler

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10209", help="Server URI")
    # data-dir больше не является обязательным
    parser.add_argument("--data-dir", default=None, help="Path to models (optional in V3, handled by auto_download)")
    parser.add_argument("--language", default="en", help="Default voice language")
    parser.add_argument("--steps", type=int, default=5, help="Denoising steps")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    parser.add_argument("--debug", action="store_true", help="Debug logs")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT, help="Log format")
    parser.add_argument("--version", action="version", version=__version__)
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)
    
    os.environ["SUPERTONIC_INTRA_OP_THREADS"] = str(args.threads)
    os.environ["SUPERTONIC_INTER_OP_THREADS"] = str(max(1, args.threads // 2))

    _LOGGER.info("Initializing Supertonic V3 engine...")
    
    engine = SupertonicEngine(steps=args.steps, speed=args.speed, model_path=args.data_dir)
    
    try:
        await asyncio.to_thread(engine.load)
    except Exception as e:
        _LOGGER.fatal(f"Load error: {e}")
        return

    wyoming_voices =[]
    
    # 31 язык (V3)
    supported_languages =[
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", 
        "el", "es", "et", "fi", "fr", "hi", "hr", "hu", 
        "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", 
        "ru", "sk", "sl", "sv", "tr", "uk", "vi"
    ]
    
    for voice_id in engine.available_voices:
        readable_name = voice_id
        if voice_id.startswith("M") and voice_id[1:].isdigit():
            readable_name = f"Male {voice_id[1:]}"
        elif voice_id.startswith("F") and voice_id[1:].isdigit():
            readable_name = f"Female {voice_id[1:]}"
            
        wyoming_voices.append(
            TtsVoice(
                name=voice_id,
                description=readable_name,
                attribution=Attribution(name="Supertone", url="https://github.com/supertone-inc/supertonic"),
                installed=True,
                version=__version__,
                languages=supported_languages,
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="Supertonic",
                description="Supertonic V3",
                attribution=Attribution(name="Supertone", url="https://huggingface.co/Supertone/supertonic-3"),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=not args.no_streaming,
                voices=wyoming_voices,
            )
        ],
    )
    
    uri = urlparse(args.uri)
    if uri.scheme != "tcp" or not uri.hostname or not uri.port:
        _LOGGER.fatal("Only tcp://HOST:PORT URI is supported")
        return

    _LOGGER.info(f"Starting Wyoming server on {uri.hostname}:{uri.port}")

    server = AsyncTcpServer(host=uri.hostname, port=uri.port)

    handler_factory = partial(
        SupertonicEventHandler,
        wyoming_info,
        args,
        engine,
    )

    try:
        await server.run(handler_factory)
    except KeyboardInterrupt:
        pass

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()