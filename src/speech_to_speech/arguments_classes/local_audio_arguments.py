import sys
from dataclasses import dataclass, field
from typing import Optional

ECHO_CANCELLATION_DOCS = "https://github.com/huggingface/speech-to-speech/blob/main/docs/echo-cancellation.md"


@dataclass
class LocalAudioArguments:
    local_audio_tool_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Importable module defining TOOLS and async execute_tool(name, arguments).",
            "aliases": ["--tool-module"],
        },
    )
    local_audio_input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index used by the local command."},
    )
    local_audio_output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice output device index used by the local command."},
    )
    local_audio_chunk_size: int = field(
        default=1024,
        metadata={"help": "Microphone and speaker callback block size in samples. Default is 1024."},
    )
    local_audio_playback_buffer_ms: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Audio to buffer before local playback starts, in milliseconds. "
                "Defaults to 196 for OpenAI-compatible TTS and 0 otherwise."
            ),
            "aliases": ["--playback-buffer-ms"],
        },
    )
    local_audio_block_mic_during_playback: bool = field(
        default=False,
        metadata={
            "help": "Pause local microphone capture while audio is playing. Disabled by default so barge-in works."
        },
    )
    local_audio_echo_cancellation: bool = field(
        default=False,
        metadata={
            "help": (
                "macOS only: cancel speaker echo with Apple's voice processing so you can interrupt by voice "
                "without headphones. Uses the system default microphone and speaker. "
                "Requires: pip install 'speech-to-speech[macos-aec]'."
            ),
            "aliases": ["--echo-cancellation"],
        },
    )
    local_audio_print_json: bool = field(
        default=False,
        metadata={"help": "Print raw Realtime events received by the packaged local audio client."},
    )

    def __post_init__(self) -> None:
        if not self.local_audio_echo_cancellation:
            return
        if sys.platform != "darwin":
            raise ValueError(
                "--local_audio_echo_cancellation is only available on macOS. On Linux, use the PulseAudio or "
                f"PipeWire echo canceller instead: {ECHO_CANCELLATION_DOCS}"
            )
        if self.local_audio_input_device is not None or self.local_audio_output_device is not None:
            raise ValueError(
                "--local_audio_echo_cancellation uses the system default microphone and speaker; "
                "drop --local_audio_input_device and --local_audio_output_device, "
                "and pick the devices in System Settings > Sound instead."
            )
        from speech_to_speech.api.openai_realtime.macos_voice_processing import (
            INSTALL_HINT,
            voice_processing_available,
        )

        if not voice_processing_available():
            raise ValueError(f"--local_audio_echo_cancellation needs PyObjC's AVFoundation bindings. {INSTALL_HINT}")
