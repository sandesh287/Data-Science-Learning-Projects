# 17. Simple Personal Assistant using Python Speech Libraries
# pip install SpeechRecognition
# pip install pyttsx3==2.90
# pip install pyaudio



# import necessary libraries
import speech_recognition as sr   # used for converting spoken words to text using Google's speech recognition API
import pyttsx3   # used for text to speech, allowing the assistant to respond vocally
import datetime
import webbrowser
import os
import time


# Initialize the speech engine for text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 175)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)   # 0=male, 1=female


# Initialize recognizer once
recognizer = sr.Recognizer()


# Function to make the assistant speak
def speak(text):
  # time.sleep(0.3)   # short pause before speaking
  print(f'Assistant: {text}')
  engine.say(text)
  engine.runAndWait()
  time.sleep(0.3)   # small delay to avoid microphone conflict


# Function to take a voice command from the user
def take_command():
  with sr.Microphone() as source:
    print('Listening...')
    recognizer.adjust_for_ambient_noise(source, duration=0.5)
    audio = recognizer.listen(source)
    
    try:
      print('Recognizing...')
      command = recognizer.recognize_google(audio)
      print(f'User said: {command}')
    except sr.UnknownValueError:
      print('Sorry, I did not understand that.')
      return None
    except sr.RequestError:
      speak('Network error.')
      return None
  
  return command.lower()


# Function to respond to different commands
def respond(command):
  if not command:
    return
  
  if 'hello' in command or 'hi' in command:
    speak('Hello! How can I assist you today?')
  
  elif 'time' in command:
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    speak(f'The current time is {current_time}')
  
  elif 'search' in command:
    speak('What would you like to search for?')
    search_query = take_command()
    if search_query:
      speak(f'Searching for {search_query}')
      webbrowser.open(f"https://www.google.com/search?q={search_query}")
  
  elif 'open' in command:
    if 'edge' in command:
      speak('Opening Microsoft Edge')
      os.system('start msedge')
    elif 'calculator' in command:
      speak('Opening Calculator')
      os.system('start calc')
  
  elif 'bye' in command or 'exit' in command or 'quit' in command or 'see you' in command:
    speak('Goodbye! Have a great day.')
    exit()
  
  else:
    speak("I'm sorry, I don't know that command.")


# Main function to run the assistant
def run_assistant():
  speak("Hello, I am your assistant. How can I help you?")
  while True:
    command = take_command()
    if command:
      respond(command)


# Start the assistant
run_assistant()