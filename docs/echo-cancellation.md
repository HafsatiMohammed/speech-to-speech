# Echo cancellation for local voice chat

By default, `speech-to-speech local` reads the microphone and plays replies
through the default PortAudio devices. It does not cancel acoustic echo itself,
so on laptop speakers the assistant can hear its own voice. The VAD then treats
that voice as a barge-in and cuts the reply off.

You have three options:

- Use headphones.
- Pass `--local_audio_block_mic_during_playback`. This stops the feedback but
  also stops you from interrupting the assistant by voice.
- Run the audio through an echo canceller. This keeps voice interruptions
  working without headphones. On macOS, pass `--echo-cancellation`.

## macOS

macOS has no built-in system-wide echo-cancelled device. Apple's echo canceller
lives in the Voice Processing I/O audio unit, which each app opens for itself;
video call apps and browsers use it. PortAudio and `sounddevice` cannot open
that unit, so `local` has an alternate audio backend for it.

### `local` with `--echo-cancellation`

Install the PyObjC bindings, then add the flag:

```bash
pip install 'speech-to-speech[macos-aec]'
speech-to-speech local --mac-optimal-settings --echo-cancellation
```

With this flag, `local` plays and records through `AVAudioEngine` with voice
processing turned on, instead of PortAudio. Everything the client plays is
removed from the microphone signal before it is sent to the server.

- The flag uses the system default microphone and speaker. Pick them in
  System Settings > Sound. It cannot be combined with
  `--local_audio_input_device` or `--local_audio_output_device`.
- On the first run, macOS asks whether your terminal may use the microphone.
- While the client is running, macOS lowers the volume of other apps' audio.
  This is standard voice-processing behavior, the same as during a FaceTime
  call.
- Up to about 130 ms of the reply that was already scheduled on the speaker
  still plays after you interrupt.

This backend is covered by unit tests with a fake `AVFoundation` module, but it
has not yet been run on a Mac. Please report what you find, including your
macOS version and audio devices.

### Browser client

If `--echo-cancellation` doesn't work on your machine, you can run the server
and talk to it from a browser. The [browser demo](../demo/README.md) asks for
the microphone with `echoCancellation: true`, so the browser cancels the
assistant's playback before the audio reaches the server:

```bash
speech-to-speech serve --mac-optimal-settings --enable_live_transcription
# in another terminal, from a source checkout
npm ci --prefix demo
uv pip install -r demo/requirements.txt
SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime \
    uv run uvicorn --app-dir demo server:app --port 7860
```

Open <http://localhost:7860/> in Chrome or Safari, allow the microphone, and talk
through the laptop speakers.
