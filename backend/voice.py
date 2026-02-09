import speech_recognition as sr
import time

def voice_recognition():
    recognizer = sr.Recognizer()
    
    try:
        # Check available microphones
        print("\nAvailable microphones:")
        mics = sr.Microphone.list_microphone_names()
        for i, name in enumerate(mics):
            print(f"{i}: {name}")
        
        # Try to use the default microphone
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... (stay silent for 2 seconds)")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            recognizer.dynamic_energy_threshold = True
            print("Microphone ready! Speak now...")
            
            while True:
                try:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                    print("Processing your speech...")
                    
                    text = recognizer.recognize_google(audio).lower()
                    print(f"You said: {text}")
                    
                    if "exit" in text:
                        print("Goodbye!")
                        break
                        
                    # Command examples
                    if "hello" in text:
                        print("Hello there!")
                    elif "time" in text:
                        print(f"Current time is {time.strftime('%H:%M')}")
                        
                except sr.WaitTimeoutError:
                    print("I didn't hear anything. Try speaking again...")
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand that.")
                except KeyboardInterrupt:
                    print("\nExiting by user request...")
                    break
                    
    except OSError as e:
        print("\nERROR: Microphone access problem!")
        print(f"Details: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure your microphone is properly connected")
        print("2. Check if another program is using the microphone")
        print("3. Try specifying a different microphone index")
        
        if mics:
            print("\nTry modifying the script to use:")
            print(f"with sr.Microphone(device_index=0) as source:  # Using first microphone")
        else:
            print("\nNo microphones found!")

if __name__ == "__main__":
    print("=== Python Voice Recognition ===")
    print("Press Ctrl+C to exit at any time\n")
    voice_recognition()