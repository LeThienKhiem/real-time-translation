"""
🌐 Real-time Speech Translator — Web Edition
Flask + Socket.IO backend for real-time streaming to browser.

Supports:
  - Microphone capture (sounddevice)
  - System audio / loopback (WASAPI via PyAudioWPatch)
  - Real-time transcription (faster-whisper)
  - Real-time translation (Google Translate)
  - AI summary of conversation fragments
"""

import collections
import threading
import time
import numpy as np
import argparse

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from audio_capture import AudioCapture
from speech_recognizer import SpeechRecognizer
from translator import Translator, LANGUAGE_NAMES, map_lang_code

# ----------------------------------------------------------------
#  Flask app
# ----------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "realtime-translator"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ----------------------------------------------------------------
#  Global state
# ----------------------------------------------------------------

state = {
    "audio": None,
    "recognizer": None,
    "translator": None,
    "is_translating": False,
    "source": "mic",
    "target_lang": "vi",
    "model_size": "base",
    "device": "cpu",
    "device_index": None,
}

audio_buffer = collections.deque()
buffer_lock = threading.Lock()

# Collected translated sentences for summarization
translated_sentences = []
sentences_lock = threading.Lock()

PROCESS_INTERVAL = 0.5   # Faster cycle = more real-time
WINDOW_SECONDS = 4.0


# ----------------------------------------------------------------
#  Routes
# ----------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------------------------------------------
#  Socket.IO events
# ----------------------------------------------------------------

@socketio.on("connect")
def on_connect():
    print(f"✅ Client connected: {request.sid}")
    emit("status", {"status": "connected", "config": {
        "source": state["source"],
        "target_lang": state["target_lang"],
        "model_size": state["model_size"],
        "is_translating": state["is_translating"],
    }})


@socketio.on("start")
def on_start(data=None):
    print(f"\n📩 START received: {data}")
    if data:
        if "source" in data:
            state["source"] = data["source"]
            print(f"   → source set to: {state['source']}")
        if "target_lang" in data:
            state["target_lang"] = data["target_lang"]
            if state["translator"]:
                state["translator"].set_target_language(data["target_lang"])
        if "model_size" in data:
            state["model_size"] = data["model_size"]
        if "device_index" in data and data["device_index"] is not None:
            state["device_index"] = int(data["device_index"])
        else:
            state["device_index"] = None

    if state["is_translating"]:
        emit("log", {"msg": "Already running"})
        return

    # Initialize recognizer
    if state["recognizer"] is None:
        emit("log", {"msg": f"Loading Whisper '{state['model_size']}'..."})
        socketio.sleep(0.1)
        state["recognizer"] = SpeechRecognizer(
            model_size=state["model_size"], device=state["device"]
        )
        emit("log", {"msg": "Model loaded!"})

    if state["translator"] is None:
        state["translator"] = Translator(target_language=state["target_lang"])

    state["audio"] = AudioCapture(
        source=state["source"],
        chunk_duration=0.2,  # Smaller chunks = lower latency
        device_index=state.get("device_index"),
    )

    state["is_translating"] = True

    # Clear sentences
    with sentences_lock:
        translated_sentences.clear()

    # Start threads
    threading.Thread(target=_audio_collector, daemon=True).start()
    threading.Thread(target=_transcription_loop, daemon=True).start()

    src_label = "System Audio (Loopback)" if state["source"] == "system" else "Microphone"
    emit("started", {"source": state["source"], "target": state["target_lang"]})
    emit("log", {"msg": f"Started — {src_label} → {LANGUAGE_NAMES.get(state['target_lang'], state['target_lang'])}"})

    try:
        state["audio"].start()
    except Exception as e:
        state["is_translating"] = False
        emit("error", {"msg": str(e)})


@socketio.on("stop")
def on_stop():
    state["is_translating"] = False
    if state["audio"]:
        try:
            state["audio"].stop()
        except Exception:
            pass
    with buffer_lock:
        audio_buffer.clear()
    emit("stopped", {})
    emit("log", {"msg": "Stopped"})


@socketio.on("change_target")
def on_change_target(data):
    lang = data.get("lang", "vi")
    state["target_lang"] = lang
    if state["translator"]:
        state["translator"].set_target_language(lang)
    emit("log", {"msg": f"Target → {LANGUAGE_NAMES.get(lang, lang)}"})


@socketio.on("list_devices")
def on_list_devices():
    """List input devices (mic via sounddevice) and output devices (loopback via soundcard)."""
    devices = []

    # Input devices (for mic mode)
    try:
        import sounddevice as sd
        default_in = sd.default.device[0]
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "is_default": (i == default_in),
                    "type": "input",
                })
    except Exception as e:
        emit("log", {"msg": f"sounddevice error: {e}"})

    # Output devices (for system audio loopback via PyAudioWPatch)
    try:
        from audio_capture import AudioCapture
        out_devs = AudioCapture.list_output_devices()
        for dev in out_devs:
            devices.append({
                "id": dev["index"],
                "name": dev["name"],
                "is_default": (dev["index"] == 0),  # first WASAPI output is usually default
                "type": "output",
            })
        if not out_devs:
            emit("log", {"msg": "No WASAPI output devices found"})
    except ImportError:
        emit("log", {"msg": "pyaudiowpatch not installed — run: pip install pyaudiowpatch"})
    except Exception as e:
        emit("log", {"msg": f"Output device error: {e}"})

    emit("devices", {"devices": devices})


@socketio.on("request_summary")
def on_request_summary():
    """Generate a summary of all translated sentences so far."""
    with sentences_lock:
        if not translated_sentences:
            emit("summary", {"text": "Chưa có nội dung để tổng hợp."})
            return
        combined = "\n".join(translated_sentences)

    # Use Google Translate to clean up fragmented text by re-translating
    # the combined text. This produces more coherent output.
    target = state["target_lang"]
    try:
        from deep_translator import GoogleTranslator
        # Translate combined back to get coherent version
        result = GoogleTranslator(source="auto", target=map_lang_code(target)).translate(combined)
        socketio.emit("summary", {"text": result or combined, "count": len(translated_sentences)})
    except Exception:
        socketio.emit("summary", {"text": combined, "count": len(translated_sentences)})


# ----------------------------------------------------------------
#  Background threads
# ----------------------------------------------------------------

def _audio_collector():
    audio = state["audio"]
    chunk_dur = audio.chunk_duration
    max_chunks = int(WINDOW_SECONDS / chunk_dur) + 5
    level_counter = 0

    while state["is_translating"]:
        try:
            chunk = audio.audio_queue.get(timeout=chunk_dur * 3)
            with buffer_lock:
                audio_buffer.append(chunk)
                while len(audio_buffer) > max_chunks:
                    audio_buffer.popleft()

            # Send audio level to UI every ~5 chunks (~1s) for volume meter
            level_counter += 1
            if level_counter % 5 == 0:
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                peak = float(np.max(np.abs(chunk)))
                socketio.emit("audio_level", {"rms": rms, "peak": peak})
        except Exception:
            continue


def _get_buffer_audio():
    with buffer_lock:
        if not audio_buffer:
            return None
        return np.concatenate(list(audio_buffer), axis=0)


def _is_new_text(new_text, old_text):
    if not old_text:
        return True
    if new_text == old_text:
        return False
    common = len(set(new_text.split()) & set(old_text.split()))
    total = max(len(new_text.split()), len(old_text.split()), 1)
    return (common / total) < 0.65


def _transcription_loop():
    last_translated = ""
    cycle = 0

    while state["is_translating"]:
        time.sleep(PROCESS_INTERVAL)
        cycle += 1

        audio_data = _get_buffer_audio()
        if audio_data is None:
            if cycle % 10 == 0:
                socketio.emit("log", {"msg": "No audio in buffer"})
            continue

        rms = float(np.sqrt(np.mean(audio_data ** 2)))

        # Log every 5 cycles so user can see what's happening
        if cycle % 5 == 0:
            buf_sec = len(audio_data) / 16000
            socketio.emit("log", {"msg": f"Buffer: {buf_sec:.1f}s | RMS: {rms:.5f}"})
            print(f"   [cycle {cycle}] buffer={buf_sec:.1f}s  rms={rms:.5f}")

        # Very low threshold
        if rms < 0.0003:
            continue

        # --- Transcribe ---
        result = state["recognizer"].transcribe(audio_data)
        text = result["text"].strip()
        lang = result["language"]

        if not text:
            socketio.emit("log", {"msg": f"RMS={rms:.4f} — Whisper returned empty"})
            print(f"   [cycle {cycle}] Whisper empty (rms={rms:.4f})")
            continue

        print(f"   [cycle {cycle}] STT: [{lang}] {text[:60]}")

        lang_name = LANGUAGE_NAMES.get(lang, lang)

        # Emit partial IMMEDIATELY
        socketio.emit("partial", {
            "text": text,
            "lang": lang,
            "lang_name": lang_name,
        })

        # --- Translate if meaningfully new ---
        if _is_new_text(text, last_translated):
            last_translated = text
            try:
                trans = state["translator"].translate(text, source_language=lang)
                translated = trans["translated"]

                # Store for summary
                with sentences_lock:
                    translated_sentences.append(translated)
                    # Auto-emit summary every 5 sentences
                    count = len(translated_sentences)
                    if count % 5 == 0:
                        combined = " ".join(translated_sentences)
                        socketio.emit("auto_summary", {
                            "text": combined,
                            "count": count,
                        })

                socketio.emit("translated", {
                    "original": text,
                    "translated": translated,
                    "source_lang": lang_name,
                    "target_lang": LANGUAGE_NAMES.get(state["target_lang"], state["target_lang"]),
                    "stt_time": result.get("duration", 0),
                    "translate_time": trans.get("duration", 0),
                })
            except Exception as e:
                socketio.emit("error", {"msg": f"Translation error: {e}"})


# ----------------------------------------------------------------
#  Main
# ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time Translator Web UI")
    parser.add_argument("--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    state["model_size"] = args.model
    state["device"] = args.device

    print("=" * 60)
    print("🌐 Real-time Speech Translator — Web UI")
    print(f"   Open http://{args.host}:{args.port} in your browser")
    print("=" * 60)

    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
