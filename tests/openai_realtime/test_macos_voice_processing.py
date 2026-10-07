import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.api.openai_realtime.audio_client as audio_client_module
from speech_to_speech.api.openai_realtime import macos_voice_processing
from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
from speech_to_speech.api.openai_realtime.macos_voice_processing import (
    VoiceProcessingAudioIO,
    float_to_pcm16,
    pcm16_to_float,
)
from speech_to_speech.arguments_classes.local_audio_arguments import LocalAudioArguments
from speech_to_speech.s2s_pipeline import parse_arguments


class _VarList:
    """Mimics the objc.varlist PyObjC returns for a float channel pointer."""

    def __init__(self, samples):
        self._samples = samples

    def as_buffer(self, count):
        return memoryview(self._samples[:count])


class FakePCMBuffer:
    def __init__(self, fmt=None, capacity=0, samples=None):
        self.format = fmt
        self.samples = np.zeros(capacity, dtype=np.float32) if samples is None else samples
        self.frame_length = len(self.samples) if samples is not None else 0

    @classmethod
    def alloc(cls):
        return cls.__new__(cls)

    def initWithPCMFormat_frameCapacity_(self, fmt, capacity):
        self.__init__(fmt, capacity)
        return self

    def setFrameLength_(self, frames):
        self.frame_length = frames

    def frameLength(self):
        return self.frame_length

    def floatChannelData(self):
        return [_VarList(self.samples)]


class FakeFormat:
    def __init__(self, rate=0.0):
        self.rate = rate

    @classmethod
    def alloc(cls):
        return cls()

    def initWithCommonFormat_sampleRate_channels_interleaved_(self, _common, rate, channels, interleaved):
        assert channels == 1 and interleaved is False
        self.rate = rate
        return self

    def sampleRate(self):
        return self.rate


class FakeNode:
    def __init__(self, name="node"):
        self.name = name
        self.volume = 1.0
        self.tap = None
        self.scheduled = []
        self.playing = False

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def setOutputVolume_(self, volume):
        self.volume = volume

    def installTapOnBus_bufferSize_format_block_(self, bus, size, fmt, block):
        self.tap = (bus, size, fmt, block)

    def removeTapOnBus_(self, bus):
        self.tap = None

    def scheduleBuffer_completionHandler_(self, buffer, handler):
        self.scheduled.append(buffer)
        handler()

    def play(self):
        self.playing = True

    def stop(self):
        self.playing = False


class FakeInputNode(FakeNode):
    def __init__(self, rate, voice_processing_ok):
        super().__init__("input")
        self.rate = rate
        self.voice_processing_ok = voice_processing_ok
        self.voice_processing = False

    def setVoiceProcessingEnabled_error_(self, enabled, _error):
        if not self.voice_processing_ok:
            return False, "unsupported"
        self.voice_processing = enabled
        return True, None

    def outputFormatForBus_(self, _bus):
        return FakeFormat(self.rate)


def make_fake_avfoundation(*, rate=48000.0, voice_processing_ok=True):
    state = SimpleNamespace(engines=[])

    class FakeEngine:
        @classmethod
        def alloc(cls):
            return cls()

        def init(self):
            self.input = FakeInputNode(rate, voice_processing_ok)
            self.main_mixer = FakeNode("main")
            self.attached = []
            self.connections = []
            self.running = False
            state.engines.append(self)
            return self

        def inputNode(self):
            return self.input

        def mainMixerNode(self):
            return self.main_mixer

        def attachNode_(self, node):
            self.attached.append(node)

        def connect_to_format_(self, source, destination, fmt):
            self.connections.append((source, destination, fmt))

        def prepare(self):
            pass

        def startAndReturnError_(self, _error):
            self.running = True
            return True, None

        def stop(self):
            self.running = False

    module = SimpleNamespace(
        AVAudioEngine=FakeEngine,
        AVAudioMixerNode=FakeNode,
        AVAudioPlayerNode=FakeNode,
        AVAudioPCMBuffer=FakePCMBuffer,
        AVAudioFormat=FakeFormat,
        AVAudioPCMFormatFloat32=1,
    )
    return module, state


def make_io(monkeypatch, *, fill_output=None, **fake_kwargs):
    module, state = make_fake_avfoundation(**fake_kwargs)
    monkeypatch.setitem(sys.modules, "AVFoundation", module)
    received = []
    io = VoiceProcessingAudioIO(
        send_rate=16000,
        recv_rate=24000,
        chunk_size=4,
        on_input=received.append,
        fill_output=fill_output or (lambda chunk: None),
    )
    return io, state.engines[0], received


def test_pcm16_conversion_round_trips_and_clips():
    samples = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    pcm = float_to_pcm16(samples)

    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 16383, -16383, 32767, -32767]
    np.testing.assert_allclose(pcm16_to_float(pcm), np.clip(samples, -1, 1), atol=1e-4)


def test_voice_processing_engine_routes_mic_through_resampling_mixer(monkeypatch):
    io, engine, _ = make_io(monkeypatch)

    assert engine.input.voice_processing is True
    capture_mixer, muted_mixer, player = engine.attached
    assert [(src, dst, getattr(fmt, "rate", None)) for src, dst, fmt in engine.connections] == [
        (engine.input, capture_mixer, 48000.0),
        (capture_mixer, muted_mixer, 16000.0),
        (muted_mixer, engine.main_mixer, None),
        (player, engine.main_mixer, 24000.0),
    ]
    assert muted_mixer.volume == 0.0
    assert capture_mixer.tap[:3] == (0, 4, None)


def test_voice_processing_input_tap_delivers_pcm16(monkeypatch):
    io, engine, received = make_io(monkeypatch)
    tap = engine.attached[0].tap[3]

    tap(FakePCMBuffer(samples=np.array([0.5, -0.5], dtype=np.float32)), None)
    tap(FakePCMBuffer(samples=np.zeros(0, dtype=np.float32)), None)

    assert [np.frombuffer(chunk, dtype="<i2").tolist() for chunk in received] == [[16383, -16383]]


def test_voice_processing_output_schedules_filled_chunks(monkeypatch):
    stop = Event()
    calls = []

    def fill_output(chunk):
        calls.append(len(chunk))
        chunk[:] = np.array([16384, -16384, 0, 32767], dtype="<i2").tobytes()
        if len(calls) >= 3:
            stop.set()

    io, engine, _ = make_io(monkeypatch, fill_output=fill_output)
    player = engine.attached[2]

    io.start()
    assert stop.wait(2.0)
    io.stop()
    io.close()

    assert engine.running is False and player.playing is False
    assert engine.attached[0].tap is None
    assert calls[:3] == [8, 8, 8]
    first = player.scheduled[0]
    assert first.frame_length == 4 and first.format.rate == 24000.0
    np.testing.assert_allclose(first.samples, [0.5, -0.5, 0.0, 32767 / 32768])


def test_voice_processing_reports_unsupported_hardware(monkeypatch):
    with pytest.raises(RuntimeError, match="voice processing: unsupported"):
        make_io(monkeypatch, voice_processing_ok=False)


def test_voice_processing_reports_missing_microphone(monkeypatch):
    with pytest.raises(RuntimeError, match="microphone access"):
        make_io(monkeypatch, rate=0.0)


def test_voice_processing_without_pyobjc_points_to_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "AVFoundation", None)

    with pytest.raises(RuntimeError, match=r"speech-to-speech\[macos-aec\]"):
        VoiceProcessingAudioIO(
            send_rate=16000, recv_rate=16000, chunk_size=4, on_input=lambda _: None, fill_output=lambda _: None
        )


async def test_audio_session_uses_voice_processing_instead_of_sounddevice(monkeypatch):
    created = []

    class FakeVoiceProcessingIO:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def start(self):
            raise RuntimeError("stop after start")

        def stop(self):
            pass

        def close(self):
            created.append("closed")

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    monkeypatch.setattr(macos_voice_processing, "VoiceProcessingAudioIO", FakeVoiceProcessingIO)

    with pytest.raises(RuntimeError, match="stop after start"):
        await audio_client_module._run_audio_session(
            SimpleNamespace(),
            RealtimeAudioClientConfig(echo_cancellation=True, recv_rate=24000, chunk_size=512),
            Event(),
        )

    kwargs = created[0]
    assert (kwargs["send_rate"], kwargs["recv_rate"], kwargs["chunk_size"]) == (16000, 24000, 512)
    assert created[1] == "closed"


def test_echo_cancellation_flag_is_rejected_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(ValueError, match="only available on macOS.*echo-cancellation.md"):
        LocalAudioArguments(local_audio_echo_cancellation=True)


def test_echo_cancellation_flag_rejects_explicit_devices(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")

    with pytest.raises(ValueError, match="system default microphone"):
        LocalAudioArguments(local_audio_echo_cancellation=True, local_audio_input_device=2)


def test_echo_cancellation_flag_requires_pyobjc(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(macos_voice_processing, "voice_processing_available", lambda: False)

    with pytest.raises(ValueError, match=r"speech-to-speech\[macos-aec\]"):
        LocalAudioArguments(local_audio_echo_cancellation=True)


@pytest.mark.parametrize("flag", ["--echo-cancellation", "--local_audio_echo_cancellation"])
def test_local_command_accepts_echo_cancellation_flag_on_macos(monkeypatch, flag):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(macos_voice_processing, "voice_processing_available", lambda: True)

    args = parse_arguments([flag], command="local")

    assert args.local_audio_kwargs.local_audio_echo_cancellation is True
    with pytest.raises(ValueError, match="not used by the HfArgumentParser"):
        parse_arguments([flag], command="serve")
