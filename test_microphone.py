import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Adjusting for background noise...")
    recognizer.adjust_for_ambient_noise(source, duration=1)

    print("Listening...")
    audio = recognizer.listen(source)

print("Processing...")

try:
    text = recognizer.recognize_google(audio)
    print(f"You said: {text}")

except sr.UnknownValueError:
    print("Sorry, I couldn't understand you.")

except sr.RequestError as e:
    print(f"Speech recognition service error: {e}")