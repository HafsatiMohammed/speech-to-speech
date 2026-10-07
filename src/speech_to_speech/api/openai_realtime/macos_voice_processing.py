"""Echo-cancelled microphone and speaker I/O for the local client on macOS.

PortAudio cannot open Apple's Voice Processing I/O audio unit, which is where
macOS keeps its acoustic echo canceller. This backend drives ``AVAudioEngine``
through PyObjC with voice processing enabled, so everything the engine plays is
subtracted from the microphone signal before it reaches the server.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.util import find_spec
from threading import Event, Semaphore, Thread
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

INSTALL_HINT = "Install it with: pip install 'speech-to-speech[macos-aec]'"

# Buffers scheduled ahead on the player node. Each holds one chunk, so this
# bounds how much already-scheduled audio still plays after a barge-in.
_QUEUED_OUTPUT_BUFFERS = 2


def voice_processing_available() -> bool:
    """Whether the PyObjC AVFoundation bindings are importable."""

    return find_spec("AVFoundation") is not None


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes | bytearray) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _channel_samples(buffer: Any, frames: int) -> np.ndarray:
    """Return a writable float32 view of channel 0 of an ``AVAudioPCMBuffer``."""

    # PyObjC exposes the channel pointer as an objc.varlist. Request enough
    # memory for either count convention (elements or bytes) and keep exactly
    # ``frames`` samples, so only the buffer's own memory is ever touched.
    view = memoryview(buffer.floatChannelData()[0].as_buffer(frames * 4)).cast("B")
    return np.frombuffer(view, dtype=np.float32, count=frames)


class VoiceProcessingAudioIO:
    """Microphone and speaker streams with the macOS echo canceller.

    Exposes the ``start``/``stop``/``close`` surface of the sounddevice streams
    it replaces. ``on_input`` receives mono PCM16 chunks at ``send_rate``.
    ``fill_output`` fills a bytearray with mono PCM16 audio at ``recv_rate``,
    like a PortAudio output callback. Both use the system default devices.
    """

    def __init__(
        self,
        *,
        send_rate: int,
        recv_rate: int,
        chunk_size: int,
        on_input: Callable[[bytes], None],
        fill_output: Callable[[bytearray], None],
    ) -> None:
        try:
            import AVFoundation
        except ImportError as exc:
            raise RuntimeError(
                f"Echo cancellation on macOS needs PyObjC's AVFoundation bindings. {INSTALL_HINT}"
            ) from exc

        self._av = AVFoundation
        self._chunk_size = chunk_size
        self._on_input = on_input
        self._fill_output = fill_output
        self._stopping = Event()
        self._free_buffers = Semaphore(_QUEUED_OUTPUT_BUFFERS)
        self._feeder: Thread | None = None
        self._tap_installed = False

        engine = AVFoundation.AVAudioEngine.alloc().init()
        input_node = engine.inputNode()
        ok, error = input_node.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            raise RuntimeError(f"Could not enable macOS voice processing: {error}")

        hardware_rate = input_node.outputFormatForBus_(0).sampleRate()
        if hardware_rate <= 0:
            raise RuntimeError(
                "No microphone is available. Check that a default input device exists and that "
                "this terminal has microphone access in System Settings > Privacy & Security."
            )

        self._input_format = self._mono_float_format(send_rate)
        self._output_format = self._mono_float_format(recv_rate)

        # Capture: a mixer converts the echo-cancelled microphone to mono at
        # send_rate. Its output feeds a muted mixer so the engine keeps
        # pulling the branch without playing the microphone back.
        capture_mixer = AVFoundation.AVAudioMixerNode.alloc().init()
        muted_mixer = AVFoundation.AVAudioMixerNode.alloc().init()
        engine.attachNode_(capture_mixer)
        engine.attachNode_(muted_mixer)
        engine.connect_to_format_(input_node, capture_mixer, self._mono_float_format(hardware_rate))
        engine.connect_to_format_(capture_mixer, muted_mixer, self._input_format)
        engine.connect_to_format_(muted_mixer, engine.mainMixerNode(), None)
        muted_mixer.setOutputVolume_(0.0)
        capture_mixer.installTapOnBus_bufferSize_format_block_(0, chunk_size, None, self._handle_input)
        self._tap_installed = True

        # Playback goes through the voice-processing output node, which gives
        # the canceller its reference signal.
        player = AVFoundation.AVAudioPlayerNode.alloc().init()
        engine.attachNode_(player)
        engine.connect_to_format_(player, engine.mainMixerNode(), self._output_format)
        engine.prepare()

        self._engine = engine
        self._capture_mixer = capture_mixer
        self._player = player

    def _mono_float_format(self, rate: float) -> Any:
        return self._av.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            self._av.AVAudioPCMFormatFloat32, float(rate), 1, False
        )

    def _handle_input(self, buffer: Any, _when: Any) -> None:
        frames = int(buffer.frameLength())
        if frames:
            self._on_input(float_to_pcm16(_channel_samples(buffer, frames)))

    def _release_output_buffer(self) -> None:
        self._free_buffers.release()

    def _feed_output(self) -> None:
        chunk = bytearray(self._chunk_size * 2)
        while not self._stopping.is_set():
            if not self._free_buffers.acquire(timeout=0.1):
                continue
            self._fill_output(chunk)
            buffer = self._av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
                self._output_format, self._chunk_size
            )
            buffer.setFrameLength_(self._chunk_size)
            _channel_samples(buffer, self._chunk_size)[:] = pcm16_to_float(chunk)
            self._player.scheduleBuffer_completionHandler_(buffer, self._release_output_buffer)

    def start(self) -> None:
        ok, error = self._engine.startAndReturnError_(None)
        if not ok:
            raise RuntimeError(f"Could not start the macOS audio engine: {error}")
        self._player.play()
        self._stopping.clear()
        self._feeder = Thread(target=self._feed_output, name="macos-voice-processing-output", daemon=True)
        self._feeder.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._feeder is not None:
            self._feeder.join(timeout=1.0)
            self._feeder = None
        self._player.stop()
        self._engine.stop()

    def close(self) -> None:
        if self._tap_installed:
            self._capture_mixer.removeTapOnBus_(0)
            self._tap_installed = False
