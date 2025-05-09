import os
import math
import threading
from flask import Flask, request, jsonify, render_template
import whisper

app = Flask(__name__)

# Load Whisper model (you can switch "base" → "small", "tiny", etc.)
model = whisper.load_model("base")

# Global dict to track progress, logs and result
progress_data = {
    "progress": 0,
    "logs": [],
    "transcript": "",
    "language": "",
    "done": False
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files.get('file')
    language = request.form.get('language', 'auto')      # 'es', 'en', or 'auto'
    task = request.form.get('task', 'transcribe')        # 'transcribe', 'translate_to_en', 'translate_to_es'

    if audio_file is None:
        return jsonify({"error": "No file received"}), 400

    # Verify extension
    filename = audio_file.filename
    extension = os.path.splitext(filename)[1].lower()
    supported_formats = ['.mp3', '.wav', '.m4a', '.mp4', '.mpeg', '.mpga', '.webm', '.ogg']
    if extension not in supported_formats:
        return jsonify({
            "error": f"Unsupported format. Allowed: {', '.join(supported_formats)}"
        }), 400

    # Save upload
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", filename)
    audio_file.save(save_path)

    # Reset progress_data
    operation = "transcription" if task == "transcribe" else "translation"
    progress_data.update({
        "progress": 0,
        "logs": [f"File received, starting {operation}..."],
        "transcript": "",
        "language": "",
        "done": False
    })

    # Launch transcription/translation in background
    thread = threading.Thread(
        target=transcribe_audio,
        args=(save_path, language, task),
        daemon=True
    )
    thread.start()

    return jsonify({"message": f"{operation.capitalize()} started."})

@app.route('/progress')
def progress():
    return jsonify(progress_data)

def transcribe_audio(path, language_option, task="transcribe"):
    try:
        # Load audio
        progress_data["logs"].append("Loading audio...")
        audio = whisper.load_audio(path)

        # Prepare 30s chunk for language detection (pad or trim)
        audio_for_lang = whisper.pad_or_trim(audio)

        audio_duration = len(audio) / whisper.audio.SAMPLE_RATE

        # Determine task type
        whisper_task = "transcribe"
        if task == "translate_to_en":
            whisper_task = "translate"
            progress_data["logs"].append("Task: Translate to English")
        elif task == "translate_to_es":
            whisper_task = "translate"
            progress_data["logs"].append("Task: Translate to Spanish")
        else:
            progress_data["logs"].append("Task: Transcribe in source language")

        # Common options
        options = {
            "task": whisper_task,
            "fp16": False,
            "verbose": False
        }

        # Auto‑detect language if requested
        if language_option == 'auto':
            progress_data["logs"].append("Detecting language automatically...")
            mel = whisper.log_mel_spectrogram(audio_for_lang).to(model.device)
            detected_language, probs = model.detect_language(mel)
            confidence = probs[detected_language].item() if hasattr(probs[detected_language], 'item') else probs[detected_language]
            progress_data["logs"].append(
                f"Detected language: {detected_language.upper()} (confidence: {confidence:.2f})"
            )
            options["language"] = detected_language
            progress_data["language"] = detected_language
        else:
            options["language"] = language_option
            progress_data["language"] = language_option
            progress_data["logs"].append(f"Using specified language: {language_option.upper()}")

        # Split audio into ~60s segments
        segment_length = 60
        num_segments = math.ceil(audio_duration / segment_length)
        progress_data["logs"].append(
            f"Audio split into {num_segments} segments (~{segment_length}s each)."
        )

        # Process each segment
        for i in range(num_segments):
            start_sample = int(i * segment_length * whisper.audio.SAMPLE_RATE)
            end_sample = int(min(audio_duration, (i+1) * segment_length) * whisper.audio.SAMPLE_RATE)
            segment = audio[start_sample:end_sample]

            progress_data["logs"].append(f"Processing segment {i+1}/{num_segments}...")
            result = model.transcribe(segment, **options)
            text = result["text"].strip()
            progress_data["transcript"] += text + " "
            progress_data["logs"].append(f"Segment {i+1} completed.")

            # Update progress percentage
            progress_data["progress"] = round((i+1) / num_segments * 100, 1)

        # Finished
        progress_data["done"] = True
        progress_data["logs"].append("Transcription complete!")
    except Exception as e:
        progress_data["logs"].append(f"Error during processing: {str(e)}")
        progress_data["done"] = True
    finally:
        # Clean up uploaded file to free memory
        try:
            os.remove(path)
            progress_data["logs"].append("Deleted uploaded file.")
        except Exception as e:
            progress_data["logs"].append(f"Failed to delete file: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
