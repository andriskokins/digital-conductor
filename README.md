# Digital Conductor

![Python](https://img.shields.io/badge/Python-100%25-blue)

Digital Conductor is an intelligent NLP-based chatbot designed to streamline train ticket booking and journey planning through natural language conversation. Users can interact with the system using plain English, eliminating the need to navigate complex web interfaces or wait in line at ticket booths.

## Introduction

Digital Conductor leverages modern Natural Language Processing techniques to understand user requests, process booking information, and answer travel-related questions. The system implements intent matching, named entity recognition, and contextual understanding to provide an intuitive booking experience.

The chatbot can:
- Process natural language requests for booking train tickets
- Recognize UK cities and correct misspellings
- Extract date and time information from various formats
- Retrieve booking information for users
- Answer common questions about train travel
- Personalize interactions based on user identity

## Features

- **Intent Recognition**: Accurately identifies user intentions using TF-IDF and cosine similarity
- **Named Entity Recognition**: Identifies cities and other travel details from user input
- **Dialogue Management**: Guides users through the booking process with appropriate prompts and confirmations
- **Error Handling**: Gracefully handles misspellings and ambiguous inputs
- **Personalization**: Remembers user names for personalised interactions
- **Performance Optimization**: Fast response times for real-time conversation

## Demo

![Demo](docs/chatbot_demo.gif)

## Installation

```bash
# Clone the repository
git clone https://github.com/andriskokins/digital-conductor.git

# Navigate to the directory
cd digital-conductor

# Install required dependencies
pip install -r requirements.txt
```

## Usage

Run the main script to start the chatbot:

```bash
python main.py
```

### Sample Interactions

```
Digital Conductor: Hello, how can I help you today?

You: I'd like to book a train ticket from London to Manchester tomorrow at 10:45am

Digital Conductor: I got the following information: From London to Manchester on 20-08-2025 10:45, is that correct?

You: yes

Digital Conductor: Ticket booked! Your reference number is 12345.
```

```
Digital Conductor: Hello, how can I help you today?

You: Can I bring my pet on the train?

Digital Conductor: Small pets are allowed if kept in a suitable carrier.
```

## Documentation

For detailed information about the project architecture, implementation details, and evaluation results, please refer to the [complete project report](docs/Chatbot_Report.pdf).

## Author

Andris Kokins - University of Nottingham
