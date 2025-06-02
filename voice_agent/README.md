# AI Health Assistant Backend

A real-time voice-enabled AI health assistant backend built with FastAPI and OpenAI's APIs for natural conversation and appointment booking.

## Overview

This backend service powers a voice-based conversational AI system that helps users book medical appointments and answers health-related questions. It features real-time voice processing, voice activity detection (VAD), and natural language understanding.

## Features

- **Real-time Voice Processing** with WebRTC VAD
- **Dual Conversation Modes**:
  - Push-to-Talk
  - Real-time (continuous listening)
- **Natural Language Processing** using OpenAI's GPT-3.5
- **Speech-to-Text** via Whisper API
- **Text-to-Speech** with realistic voices
- **Appointment Booking System**
- **Memory Management** for context retention
- **Error Recovery** with fallback responses

## Architecture

The system uses a WebSocket-based architecture for real-time bidirectional communication:

```
Client <-> WebSocket <-> FastAPI Server <-> OpenAI APIs
```

## Technical Stack

- **FastAPI**: Web framework and WebSocket handling
- **OpenAI APIs**:
  - Whisper: Speech-to-Text
  - GPT-3.5: Natural Language Processing
  - TTS: Text-to-Speech
- **WebRTC VAD**: Voice Activity Detection
- **Python 3.8+**: Core language

## Installation

### Prerequisites

1. Python 3.8 or higher
2. OpenAI API key
3. Virtual environment (recommended)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

#### requirements.txt

```txt
fastapi
uvicorn
python-dotenv
openai
httpx
webrtcvad
websockets
python-multipart
```

### Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## WebSocket API

### WebSocket Endpoint

The WebSocket server can be accessed at:

`ws://<serverurl>/voice_agent/voice`

Use this endpoint to establish a real-time bidirectional audio stream between the client and the FastAPI backend.

Example connection from the client:

````javascript
const ws = new WebSocket("ws://localhost:8000/voice_agent/voice");
````

#### Connection Modes

1. **Push-to-Talk Mode**

   ```json
   {
     "event": "set_mode",
     "mode": "push-to-talk"
   }
   ```

2. **Real-time Mode**
   ```json
   {
     "event": "set_mode",
     "mode": "real-time"
   }
   ```

#### Message Types

1. **Initial Connection**

   ```json
   {
     "event": "initial_message",
     "mode": "push-to-talk"
   }
   ```

2. **Audio Data**

   - Format: Raw PCM audio bytes
   - Sample Rate: 16kHz
   - Channels: 1 (mono)
   - Bit Depth: 16-bit

3. **End of Speech**
   ```json
   {
     "event": "end_of_turn"
   }
   ```

#### Server Responses

1. **Chat Response**

   ```json
   {
     "type": "chat_response",
     "content": "Hello! How can I assist you today?"
   }
   ```

2. **Transcript**

   ```json
   {
     "type": "chat_response",
     "transcript": "I would like to book an appointment"
   }
   ```

3. **Error**
   ```json
   {
     "type": "error",
     "content": "Error processing audio"
   }
   ```
4. **Background Noise**
   ```json
   {
     "event": "chat_noise",
     "content": "I notice there's some background noise. I'll switch to push-to-talk mode to help us communicate more clearly"
   }
   ```

## Voice Activity Detection

The system uses an enhanced VAD processor with the following parameters:

- Sample Rate: 16kHz
- Frame Duration: 30ms
- Minimum Speech Duration: 0.8s
- Minimum RMS Threshold: 300.0
