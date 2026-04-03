"""
Audio Capture Module
Supports:
  - Microphone input (sounddevice)
  - System Audio / Loopback (PyAudioWPatch — WASAPI loopback)
    → Captures audio DIRECTLY from output device (AirPods, speakers, etc.)
      WITHOUT needing a microphone. Hears exactly what the speaker plays.
"""

import numpy as np
import queue
import threading
import sys
import time


class AudioCapture:
    """Captures audio from microphone or system audio (loopback)."""

    def __init__(self, source="mic", sample_rate=16000, chunk_duration=0.2, device_index=None):
        self.source = source
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self._stream = None
        self._thread = None
        self._pyaudio = None  # for system audio
        self.device_index = device_index

    # ----------------------------------------------------------------
    #  Device discovery
    # ----------------------------------------------------------------

    @staticmethod
    def list_all_devices():
        """List all audio devices."""
        print("\n🔊 Audio Devices")
        print("=" * 65)

        # Input devices (sounddevice)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            print("\n📥 INPUT devices (Microphone):")
            default_in = sd.default.device[0]
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    tag = " ⭐" if i == default_in else ""
                    print(f"   [{i}] {dev['name']}{tag}")
        except Exception as e:
            print(f"   sounddevice error: {e}")

        # Output devices / loopback (PyAudioWPatch)
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            print("\n📢 OUTPUT devices (for System Audio loopback):")
            wasapi_info = None
            for i in range(p.get_host_api_count()):
                info = p.get_host_api_info_by_index(i)
                if "WASAPI" in info["name"]:
                    wasapi_info = info
                    break

            if wasapi_info:
                default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
                out_index = 0
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev["hostApi"] == wasapi_info["index"] and dev["maxOutputChannels"] > 0:
                        tag = " ⭐ DEFAULT" if i == default_out_idx else ""
                        print(f"   [{out_index}] {dev['name']} ({int(dev['defaultSampleRate'])}Hz){tag}")
                        out_index += 1
            else:
                print("   ⚠️  No WASAPI host API found")

            p.terminate()
        except ImportError:
            print("\n   ⚠️  pyaudiowpatch not installed")
            print("   Install: pip install pyaudiowpatch")
        except Exception as e:
            print(f"   pyaudiowpatch error: {e}")

        print()

    @staticmethod
    def list_output_devices():
        """Return list of output device dicts for the web UI."""
        devices = []
        try:
            import pyaudiowpatch as pyaudio
            p = pyaudio.PyAudio()
            wasapi_info = None
            for i in range(p.get_host_api_count()):
                info = p.get_host_api_info_by_index(i)
                if "WASAPI" in info["name"]:
                    wasapi_info = info
                    break

            if wasapi_info:
                out_index = 0
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev["hostApi"] == wasapi_info["index"] and dev["maxOutputChannels"] > 0:
                        is_loopback = dev.get("isLoopbackDevice", False)
                        devices.append({
                            "index": out_index,
                            "pa_index": i,
                            "name": dev["name"],
                            "sampleRate": int(dev["defaultSampleRate"]),
                            "channels": dev["maxOutputChannels"],
                            "isLoopback": is_loopback,
                        })
                        out_index += 1
            p.terminate()
        except ImportError:
            pass
        except Exception as e:
            print(f"   Error listing output devices: {e}")
        return devices

    # ----------------------------------------------------------------
    #  Microphone capture (sounddevice)
    # ----------------------------------------------------------------

    def _start_mic(self):
        """Start microphone capture using sounddevice."""
        import sounddevice as sd

        device_id = self._find_input_device()
        device_info = sd.query_devices(device_id)
        device_sr = int(device_info["default_samplerate"])
        device_ch = max(device_info["max_input_channels"], 1)
        target_sr = self.sample_rate

        def callback(indata, frames, time_info, status):
            audio = indata.copy()
            if device_ch > 1:
                audio = np.mean(audio, axis=1, keepdims=True)
            if device_sr != target_sr:
                mono = audio.flatten()
                new_len = int(len(mono) * target_sr / device_sr)
                resampled = np.interp(
                    np.linspace(0, len(mono) - 1, new_len),
                    np.arange(len(mono)), mono
                ).astype(np.float32)
                audio = resampled.reshape(-1, 1)
            self.audio_queue.put(audio)

        native_blocksize = int(device_sr * self.chunk_duration)
        self._stream = sd.InputStream(
            device=device_id, channels=device_ch, samplerate=device_sr,
            blocksize=native_blocksize, dtype=np.float32, callback=callback,
        )
        self._stream.start()
        print(f"🎤 Microphone: [{device_id}] {device_info['name']}")
        print(f"   {device_sr}Hz {device_ch}ch → {target_sr}Hz mono")

    def _find_input_device(self):
        import sounddevice as sd
        if self.device_index is not None:
            return self.device_index
        try:
            d = sd.default.device[0]
            if d is not None and d >= 0:
                info = sd.query_devices(d)
                if info["max_input_channels"] > 0:
                    return d
        except Exception:
            pass
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                return i
        raise RuntimeError("No input audio device found!")

    # ----------------------------------------------------------------
    #  System Audio capture (PyAudioWPatch WASAPI loopback)
    # ----------------------------------------------------------------

    def _start_system_audio(self):
        """
        Capture system audio using PyAudioWPatch's WASAPI loopback.
        This records what the OUTPUT device (AirPods/speakers) is playing,
        without needing a microphone.
        """
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            raise RuntimeError(
                "pyaudiowpatch library is required for system audio!\n"
                "Install: pip install pyaudiowpatch"
            )

        p = pyaudio.PyAudio()
        self._pyaudio = p

        # Find WASAPI host API
        wasapi_info = None
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break

        if wasapi_info is None:
            p.terminate()
            raise RuntimeError("WASAPI host API not found!")

        # Find the target output device
        target_device = None

        if self.device_index is not None:
            # Find by our list index
            out_devices = self.list_output_devices()
            if self.device_index < len(out_devices):
                target_device_info = out_devices[self.device_index]
                target_device = p.get_device_info_by_index(target_device_info["pa_index"])

        if target_device is None:
            # Use WASAPI default output
            default_idx = wasapi_info.get("defaultOutputDevice", -1)
            if default_idx >= 0:
                target_device = p.get_device_info_by_index(default_idx)

        if target_device is None:
            p.terminate()
            raise RuntimeError("No output device found for loopback!")

        # Find the loopback device for this output
        # PyAudioWPatch adds loopback devices with isLoopbackDevice=True
        loopback_device = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                # Match by name (loopback device name contains the output device name)
                if target_device["name"] in dev["name"] or dev["name"] in target_device["name"]:
                    loopback_device = dev
                    break

        # If no name match, try to find ANY loopback device
        if loopback_device is None:
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False) and dev["maxInputChannels"] > 0:
                    loopback_device = dev
                    break

        # If still no dedicated loopback device, use the output device directly
        # PyAudioWPatch allows opening output devices as loopback inputs
        if loopback_device is None:
            loopback_device = target_device
            print(f"   ℹ️  No dedicated loopback device found, using output device directly")

        device_sr = int(loopback_device["defaultSampleRate"])
        device_ch = max(loopback_device.get("maxInputChannels", 0),
                       loopback_device.get("maxOutputChannels", 2))
        if device_ch < 1:
            device_ch = 2

        print(f"🔊 System Audio (WASAPI Loopback)")
        print(f"   Output device: {target_device['name']}")
        print(f"   Loopback device: {loopback_device['name']}")
        print(f"   {device_sr}Hz, {device_ch}ch → {self.sample_rate}Hz mono")

        target_sr = self.sample_rate
        frames_per_buffer = int(device_sr * self.chunk_duration)

        def loopback_callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32).copy()

            # Reshape to (frames, channels)
            if device_ch > 1:
                audio = audio.reshape(-1, device_ch)
                mono = np.mean(audio, axis=1)
            else:
                mono = audio

            # Clamp
            np.clip(mono, -1.0, 1.0, out=mono)

            # Resample to 16kHz
            if device_sr != target_sr:
                new_len = int(len(mono) * target_sr / device_sr)
                mono = np.interp(
                    np.linspace(0, len(mono) - 1, new_len),
                    np.arange(len(mono)), mono
                ).astype(np.float32)

            self.audio_queue.put(mono.reshape(-1, 1))
            return (None, pyaudio.paContinue)

        try:
            self._stream = p.open(
                format=pyaudio.paFloat32,
                channels=device_ch,
                rate=device_sr,
                input=True,
                input_device_index=loopback_device["index"],
                frames_per_buffer=frames_per_buffer,
                stream_callback=loopback_callback,
            )
            self._stream.start_stream()
            print(f"   ✅ Loopback active!")
        except Exception as e:
            p.terminate()
            self._pyaudio = None
            raise RuntimeError(
                f"Cannot start WASAPI loopback: {e}\n"
                "Try selecting a different output device."
            )

    # ----------------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------------

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        try:
            if self.source == "system":
                self._start_system_audio()
            else:
                self._start_mic()
        except Exception as e:
            self.is_recording = False
            print(f"\n❌ Audio capture failed: {e}")
            if self.source == "system":
                print("💡 Install: pip install pyaudiowpatch")
            else:
                print("💡 Check microphone in Windows Sound Settings")
            raise

    def stop(self):
        self.is_recording = False
        if self._stream:
            try:
                if hasattr(self._stream, 'stop_stream'):
                    # PyAudio stream
                    self._stream.stop_stream()
                    self._stream.close()
                else:
                    # sounddevice stream
                    self._stream.stop()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
            except Exception:
                pass
            self._pyaudio = None
        # Loopback thread will exit on its own (checks self.is_recording)
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        print("⏹️  Audio capture stopped.")

    def get_audio_chunk(self, duration=3.0):
        chunks = []
        num_chunks = int(duration / self.chunk_duration)
        for _ in range(num_chunks):
            try:
                chunks.append(self.audio_queue.get(timeout=self.chunk_duration * 2))
            except queue.Empty:
                break
        if not chunks:
            return np.zeros((self.chunk_size, 1), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def get_audio_with_vad(self, silence_threshold=0.01, min_duration=1.0, max_duration=10.0):
        chunks = []
        speech_detected = False
        silence_count = 0
        max_silence = int(1.5 / self.chunk_duration)
        start = time.time()
        while self.is_recording:
            if time.time() - start >= max_duration:
                break
            try:
                chunk = self.audio_queue.get(timeout=self.chunk_duration * 2)
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms > silence_threshold:
                    speech_detected = True
                    silence_count = 0
                    chunks.append(chunk)
                elif speech_detected:
                    silence_count += 1
                    chunks.append(chunk)
                    if silence_count >= max_silence and (time.time() - start) >= min_duration:
                        break
                else:
                    chunks.append(chunk)
                    if len(chunks) > int(0.5 / self.chunk_duration):
                        chunks.pop(0)
            except queue.Empty:
                continue
        if not chunks:
            return np.zeros((self.chunk_size, 1), dtype=np.float32), False
        return np.concatenate(chunks, axis=0), speech_detected
