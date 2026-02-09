from flask import Flask, request, jsonify
import os
import speech_recognition as sr

app = Flask(__name__, static_folder='staticfiles')

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Voice recognition API endpoints
@app.route('/api/voice/microphones/', methods=['GET'])
def get_microphones():
    """Get a list of available microphones"""
    try:
        # Get all available microphones
        mic_names = sr.Microphone.list_microphone_names()
        mics = [{"index": i, "name": name} for i, name in enumerate(mic_names)]
        return jsonify({"microphones": mics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice/recognize/', methods=['POST'])
def recognize_speech():
    """Recognize speech from the microphone"""
    recognizer = sr.Recognizer()
    
    try:
        # Get microphone index if provided
        mic_index = None
        if 'microphone_index' in request.form:
            mic_index = int(request.form['microphone_index'])
        
        # Use the specified or default microphone
        with sr.Microphone(device_index=mic_index) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for speech
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Recognize speech using Google Speech Recognition
            text = recognizer.recognize_google(audio)
            
            return jsonify({
                "text": text,
                "confidence": 0.9  # Google doesn't provide confidence scores
            })
    
    except sr.WaitTimeoutError:
        return jsonify({"error": "No speech detected"}), 400
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition request failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 