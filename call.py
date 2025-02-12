import os
import json
import time
import base64
import threading
import sounddevice as sd
import numpy as np
from websockets.sync.client import connect  # Synchronous API
import signal
API_WS_URL = "ws://localhost:8080/call"  # Adjust if needed

# Audio settings
SAMPLE_RATE = 24000
CHANNELS = 1
DEVICE_INDEX = 0
CHUNK_DURATION = 8.0  # Shorter duration for real-time response
ws_global = None

# Signal handler to gracefully close the WebSocket connection.
def signal_handler(sig, frame):
    global ws_global
    print("\nKeyboardInterrupt received. Closing WebSocket connection...")
    if ws_global is not None:
        try:
            ws_global.close()
            print("WebSocket connection closed.")
        except Exception as e:
            print(f"Error closing WebSocket: {e}")
    sys.exit(0)

# Register the signal handler for SIGINT (Control-C)
signal.signal(signal.SIGINT, signal_handler)
def play_greeting():
    """Plays an audio greeting.
       This example uses espeak on Linux. Adjust as needed."""
    os.system('espeak "Call is connected! Continue after the beep."')

def record_audio_chunk(duration_seconds=CHUNK_DURATION):
    """Records audio and returns base64‑encoded PCM16 bytes."""
    try:
        print(f"🎙️ Recording for {duration_seconds} seconds…")
        audio_data = sd.rec(
            int(duration_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            device=DEVICE_INDEX
        )
        sd.wait()  # Block until recording finishes
        audio_data = np.clip(audio_data, -1.0, 1.0)
        int16_data = (audio_data * 32767).astype(np.int16)
        return base64.b64encode(int16_data.tobytes()).decode("ascii")
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return None

def play_audio(audio_bytes):
    """Plays back received PCM16 audio from GPT."""
    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / np.iinfo(np.int16).max
        audio_array = audio_array.reshape(-1, CHANNELS)
        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"❌ Error playing audio: {e}")

def play_beep():
    """Plays a short beep to signal the start of recording."""
    frequency = 1000  # Hz
    duration = 0.2    # seconds
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    sd.play(beep, samplerate=SAMPLE_RATE)
    sd.wait()

def generate_tick(duration=0.05, frequency=1000):
    """Generates a single tick sound."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    tick = 0.8 * np.sin(2 * np.pi * frequency * t)
    return tick.astype(np.float32)

def play_waiting_tune():
    """
    Plays a 'tick‑tick‑tick' waiting tune for 2 seconds.
    The pattern is: a 50ms tick followed by 150ms silence, repeated.
    """
    total_duration = 2.0  # seconds
    tick_duration = 0.05  # seconds
    silence_duration = 0.15  # seconds
    tick = generate_tick(tick_duration, frequency=1000)
    silence = np.zeros(int(SAMPLE_RATE * silence_duration), dtype=np.float32)
    pattern = np.concatenate((tick, silence))
    num_reps = int(np.ceil(total_duration / (len(pattern) / SAMPLE_RATE)))
    tune = np.tile(pattern, num_reps)
    tune = tune[:int(SAMPLE_RATE * total_duration)]
    sd.play(tune, samplerate=SAMPLE_RATE)
    time.sleep(total_duration)
    sd.stop()

def dump_transcript(transcript):
    """Appends the given transcript text to a file continuously."""
    if transcript:
        print(f"Dumping transcript: {transcript}")
        with open("transcript.txt", "a") as f:
            f.write(transcript + "\n")

def process_conversation(ws):
    """
    One conversation round (synchronous):
      1. Play beep.
      2. Record and send user audio.
      3. Explicitly commit audio and request GPT response.
      4. While waiting for GPT's response, run waiting tune in a separate thread.
      5. Collect GPT response (audio and text) and dump transcript continuously.
      6. Play GPT's audio.
    Returns True if successful; otherwise False.
    """
    # Step 1: Play beep
    play_beep()

    # Step 2: Record and send user audio.
    user_audio = record_audio_chunk(CHUNK_DURATION)
    if not user_audio:
        return False
    ws.send(json.dumps({
        "event": "media",
        "media": {"payload": user_audio},
        "last_chunk": True  # Signal that this is the final audio chunk.
    }))
    print("📤 Sent audio")

    # Step 2.1: Explicitly commit audio buffer and request GPT response.
    ws.send(json.dumps({"event": "commit_audio"}))
    print("✅ Committed input audio buffer.")
    ws.send(json.dumps({"event": "create_response"}))
    print("✅ Requested response from GPT.")

    # Step 3: Start waiting tune in a separate thread.
    waiting_thread = threading.Thread(target=play_waiting_tune)
    waiting_thread.start()

    # Step 4: Collect GPT response.
    audio_chunks = []
    final_transcript = None
    while True:
        try:
            message = ws.recv()
        except Exception as e:
            print("❌ Error or connection closed while waiting for response")
            waiting_thread.join()
            return False
        data = json.loads(message)
        event_type = data.get("event")
        if event_type == "media":
            audio_chunks.append(base64.b64decode(data["media"]["payload"]))
        elif event_type == "text":
            final_transcript = data.get("text", "")
            print(f"\n💬 GPT Response: {final_transcript}\n")
            dump_transcript(final_transcript)
        elif event_type == "mark" and data.get("mark", {}).get("name") == "response_end":
            break

    waiting_thread.join()

    # Step 5: Play GPT's audio.
    if audio_chunks:
        full_audio = b"".join(audio_chunks)
        play_audio(full_audio)

    time.sleep(0.5)  # Small pause before next round.
    return True

def main():
    """Establishes one persistent connection; does NOT auto-reconnect if the connection is closed."""
    global ws_global
    try:
        with connect(API_WS_URL) as ws:
            ws_global = ws
            print("\n🚀 Connected! Conversation started. Speak freely; the call will end when you close the client.\n")
            play_greeting()
            # Process conversation rounds until one fails.
            while True:
                success = process_conversation(ws)
                if not success:
                    break
    except (KeyboardInterrupt, Exception) as e:
        print(f"\n👋 Call ended: {e}")
    finally:
        ws_global = None

if __name__ == "__main__":
    import threading
    from websockets.sync.client import connect
    main()
