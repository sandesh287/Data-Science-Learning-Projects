# 3. Simple Chatbot using predefined responses



# import the regular expression module to handle pattern matching
import re


# A dictionary that maps keywords to predefined responses
responses = {
  "hello": "Hi there! How can I assist you today?",
  "hi": "Hello! How can I help you?",
  "how are you": "I'm just a bot, but I'm doing great! How about you?",
  "what is your name": "I'm a chatbot created to assist you.",
  "help": "SUre, I'm here to help. What do you need assistance with?",
  "bye": "Goodbye! Have a great day!",
  "thank you": "You're welcome! I'm happy to help.",
  "default": "I'm not sure I understand. Could you please rephrase?"
}


# Making input without punctuation and without uppercase letters
def clean_input(text):
  text = text.lower()   # convert to lowercase to make case-insensitive
  text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
  return text


# Function to find the appropriate response based on the user's input
def chatbot_response(user_input):
  
  for keyword in responses:
    if re.search(keyword, user_input):
      return responses[keyword]
  
  return responses['default']


# Main function to run the chatbot
def chatbot():
  print("Chatbot: Hello! I'm here to assist you. (type 'bye' to exit)")
  
  while True:
    # Get user input
    user_input = input('You: ')
    
    cleaned_input = clean_input(user_input)
    
    # If user types 'bye', exit the loop
    if cleaned_input == 'bye':
      print('Chatbot: Goodbye! Have a great day.')
      break
    
    # Get chatbot's response based on user input
    response = chatbot_response(cleaned_input)
    
    # print chatbot's response
    print(f'Chatbot: {response}')


# Run the chatbot
chatbot()