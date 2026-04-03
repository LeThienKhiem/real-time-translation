"""
Diagnostic script — run this to check what's working and what's not.
Usage: python diagnose.py
"""
import sys
import time
import numpy as np

print("=" * 60)
print("🔍 DIAGNOSTICS")
print("=" * 60)

# 1. Check imports
print("\n[1/5] Checking libraries...")
libs = {}
for lib in ["sounddevice", "pyaudiowpatch", "numpy", "faster_whisper", "deep_translator", "flask", "flask_socketio"]:
    try:
        __import__(lib)
        libs[lib] = True
        print(f"   ✅ {lib}")
    except ImportError:
        libs[lib] = False
        print(f"   ❌ {lib} — NOT INSTALLED")

# 2. Check audio input devices
print("\n[2/5] Audio INPUT devices (sounddevice)...")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    input_count = 0
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " ← DEFAULT" if i == sd.default.device[0] else ""
            print(f"   [{i}] {dev['name']} (SR: {int(dev['default_samplerate'])}Hz, CH: {dev['max_input_channels']}){default}")
            input_count += 1
    print(f"   Total: {input_count} input devices")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check audio output devices (for loopback)
print("\n[3/5] Audio OUTPUT devices (PyAudioWPatch WASAPI loopback)...")
if libs.get("pyaudiowpatch"):
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
            print(f"   WASAPI Host API found (index {wasapi_info['index']})")
            default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
            out_count = 0
            loopback_count = 0
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev["hostApi"] == wasapi_info["index"]:
                    is_lb = dev.get("isLoopbackDevice", False)
                    if dev["maxOutputChannels"] > 0 or is_lb:
                        default = " ← DEFAULT" if i == default_out_idx else ""
                        lb_tag = " [LOOPBACK]" if is_lb else ""
                        ch = dev.get("maxInputChannels", 0) or dev.get("maxOutputChannels", 0)
                        print(f"   [{i}] {dev['name']} (SR: {int(dev['defaultSampleRate'])}Hz, CH: {ch}){default}{lb_tag}")
                        if is_lb:
                            loopback_count += 1
                        else:
                            out_count += 1
            print(f"   Total: {out_count} output + {loopback_count} loopback devices")
        else:
            print("   ❌ No WASAPI host API found!")

        p.terminate()
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("   ⚠️  pyaudiowpatch not installed, skipping")
    print("   Run: pip install pyaudiowpatch")

# 4. Test mic recording
print("\n[4/5] Testing MICROPHONE recording (3 seconds)...")
try:
    import sounddevice as sd

    default_in = sd.default.device[0]
    if default_in is None or default_in < 0:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                default_in = i
                break

    info = sd.query_devices(default_in)
    sr = int(info["default_samplerate"])
    ch = info["max_input_channels"]
    print(f"   Using device [{default_in}]: {info['name']} ({sr}Hz, {ch}ch)")
    print(f"   Recording 3 seconds... speak now!")

    audio = sd.rec(int(3 * sr), samplerate=sr, channels=1, device=default_in, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    print(f"   ✅ Recorded! RMS={rms:.6f}  Peak={peak:.6f}")
    if rms < 0.001:
        print(f"   ⚠️  Very low audio — mic might be muted or too far")
    elif rms < 0.01:
        print(f"   ⚠️  Low audio — should work but may be inconsistent")
    else:
        print(f"   ✅ Good audio level!")
except Exception as e:
    print(f"   ❌ Mic test FAILED: {e}")

# 5. Test loopback recording
print("\n[5/5] Testing SYSTEM AUDIO loopback (3 seconds)...")
print("   ▶ Play some audio/music/video NOW before pressing Enter!")
input("   Press Enter when audio is playing... ")

if libs.get("pyaudiowpatch"):
    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()

        # Find WASAPI
        wasapi_info = None
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                wasapi_info = info
                break

        if not wasapi_info:
            print("   ❌ No WASAPI host API!")
        else:
            # Find default output
            default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
            target = p.get_device_info_by_index(default_out_idx) if default_out_idx >= 0 else None

            # Find loopback device
            loopback = None
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice", False) and target and target["name"] in dev["name"]:
                    loopback = dev
                    break

            if loopback is None:
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev.get("isLoopbackDevice", False) and dev["maxInputChannels"] > 0:
                        loopback = dev
                        break

            if loopback is None and target:
                loopback = target
                print(f"   ℹ️  No dedicated loopback device, using output device directly")

            if loopback:
                sr = int(loopback["defaultSampleRate"])
                ch = max(loopback.get("maxInputChannels", 0), loopback.get("maxOutputChannels", 2))
                if ch < 1:
                    ch = 2
                print(f"   Output: {target['name'] if target else 'unknown'}")
                print(f"   Loopback: {loopback['name']} ({sr}Hz, {ch}ch)")
                print(f"   Recording 3 seconds...")

                frames = []
                def callback(in_data, frame_count, time_info, status):
                    frames.append(np.frombuffer(in_data, dtype=np.float32).copy())
                    return (None, pyaudio.paContinue)

                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=sr,
                    input=True,
                    input_device_index=loopback["index"],
                    frames_per_buffer=int(sr * 0.1),
                    stream_callback=callback,
                )
                stream.start_stream()
                time.sleep(3)
                stream.stop_stream()
                stream.close()

                if frames:
                    audio = np.concatenate(frames)
                    if ch > 1:
                        audio = audio.reshape(-1, ch)
                        mono = np.mean(audio, axis=1)
                    else:
                        mono = audio
                    rms = float(np.sqrt(np.mean(mono ** 2)))
                    peak = float(np.max(np.abs(mono)))
                    print(f"   ✅ Recorded! RMS={rms:.6f}  Peak={peak:.6f}")
                    if rms < 0.001:
                        print(f"   ❌ No audio captured — try a different output device")
                    elif rms < 0.01:
                        print(f"   ⚠️  Low but detectable audio")
                    else:
                        print(f"   ✅ System audio captured successfully!")
                else:
                    print(f"   ❌ No frames recorded")
            else:
                print(f"   ❌ No loopback device found")

        p.terminate()
    except Exception as e:
        print(f"   ❌ Loopback test FAILED: {e}")
else:
    print("   ⚠️  pyaudiowpatch not installed — cannot test loopback")
    print("   Run: pip install pyaudiowpatch")

print("\n" + "=" * 60)
print("Done! Send the output above so I can diagnose the issue.")
print("=" * 60)
