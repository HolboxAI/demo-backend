import os
import json
from dotenv import load_dotenv
import io
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import httpx
import webrtcvad
import time
import array
import time
import array
import webrtcvad
import math

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY env var not set")

# Shared OpenAI client
async_client = AsyncOpenAI(api_key=api_key)

# Update the model settings
WHISPER_MODEL = "whisper-1"  # Best for voice detection
GPT_MODEL = "gpt-3.5-turbo"  # Fast responses
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"  # Professional female voice

# Tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_memory",
            "description": "Saves user booking information internally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Patient's full name"},
                    "phone": {"type": "string", "description": "Contact number"},
                    "date": {"type": "string", "description": "Appointment date"},
                    "time": {"type": "string", "description": "Appointment time"}
                },
                "required": ["name", "phone", "date", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_booking",
            "description": "Submit final booking to server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "phone": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string"}
                },
                "required": ["name", "phone", "date", "time"],
            },
        },
    }
]

SYSTEM_PROMPT = """
System settings:
Tool use: enabled.

Instructions:
- You are an AI assistant named Nova, primarily focused on healthcare but can help with general questions too.
- Your main responsibilities:
  1. Help users book hospital appointments
  2. Answer health-related questions
  3. Respond to general questions about:
     - Basic facts and information
     - Personal introductions and greetings

- For appointment booking, collect:
  - Full name
  - Phone number
  - Preferred date and time
- Confirm details before submitting
- Allow users to update information if needed
- For medical emergencies, advise immediate professional help

Response Guidelines:
- Keep responses clear and concise
- Use conversational, friendly tone
- If you don't know something, be honest about it
- For basic questions, give direct answers
- For health questions, be informative but remind users this isn't medical advice
- Handle errors gracefully with friendly messages

Example responses:
- "I didn't catch that clearly. Could you rephrase your question?"
- "While I can't give medical advice, I can explain what that term means..."
- "I'd be happy to help you book an appointment to discuss this with a doctor."
"""


class EnhancedVADProcessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: int = 30,      # ms
        min_duration: float = 0.6,     # seconds of continuous speech to emit
        min_rms: float = 350.0,        # RMS threshold for “human‐level” speaking
    ):
        if sample_rate not in [8000, 16000, 24000, 32000, 48000]:
            raise ValueError("Invalid sample rate for VAD")

        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)  # samples per frame
        self.frame_bytes = self.frame_size * 2  # 16‐bit PCM → 2 bytes/sample
        self.min_duration = min_duration
        self.min_frames = int((min_duration * 1000) / frame_duration)
        self.min_rms = min_rms

        # Internal state:
        self.buffer = b""
        self.speech_buffer = b""
        self.is_speaking = False
        self.last_voice_time = 0.0
        self.silence_frames = 0

    def _compute_rms(self, pcm_frame: bytes) -> float:
        """
        Compute the RMS (root-mean-square) of a 16-bit PCM frame.
        """
        # Interpret bytes as signed 16‐bit little‐endian samples
        shorts = array.array('h', pcm_frame)
        if len(shorts) == 0:
            return 0.0
        # Compute RMS = sqrt(mean(x_i^2))
        sum_squares = 0
        for s in shorts:
            sum_squares += (s * s)
        mean_square = sum_squares / len(shorts)
        return math.sqrt(mean_square)

    def process(self, audio_data: bytes) -> bytes | None:
        """
        Feed in raw 16‐bit PCM data (mono, little‐endian). Returns a
        concatenated speech segment (>= min_duration in length) if detected;
        otherwise returns None.
        """
        self.buffer += audio_data
        segments = []

        # Keep slicing off full‐length frames.
        while len(self.buffer) >= self.frame_bytes:
            frame = self.buffer[: self.frame_bytes]
            self.buffer = self.buffer[self.frame_bytes :]

            # 1) Check RMS energy first. If below threshold, treat as silence:
            rms = self._compute_rms(frame)
            if rms < self.min_rms:
                # Too quiet → count as silence
                if self.is_speaking:
                    self.speech_buffer += frame
                    self.silence_frames += 1
                    # If we have been “quiet” for long enough, end the current speech turn:
                    if (
                        time.time() - self.last_voice_time > 1.8
                        or self.silence_frames > 19
                    ):
                        num_frames = len(self.speech_buffer) // self.frame_bytes
                        if num_frames >= self.min_frames:
                            segments.append(self.speech_buffer)
                        # Reset
                        self.speech_buffer = b""
                        self.is_speaking = False
                        self.silence_frames = 0
                # If we’re not in a speech turn, simply skip
                continue

            # 2) If RMS≥threshold, consult the WebRTC-VAD
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception as e:
                is_speech = False

            if is_speech:
                # Reset silence counter, accumulate into buffer
                self.speech_buffer += frame
                self.last_voice_time = time.time()
                self.silence_frames = 0
                if not self.is_speaking:
                    self.is_speaking = True
            else:
                # VAD says “non‐speech” but we only add it if we are in a speech turn,
                # to give some padding.
                if self.is_speaking:
                    self.speech_buffer += frame
                    self.silence_frames += 1
                    # End of speech turn if prolonged silence
                    if (
                        time.time() - self.last_voice_time > 0.3
                        or self.silence_frames > 5
                    ):
                        num_frames = len(self.speech_buffer) // self.frame_bytes
                        if num_frames >= self.min_frames:
                            segments.append(self.speech_buffer)
                        self.speech_buffer = b""
                        self.is_speaking = False
                        self.silence_frames = 0

        # If any valid segments are found, return their concatenation.
        if segments:
            return b"".join(segments)
        else:
            return None



def is_unwanted_transcript(text: str) -> bool:
    """Check if transcript contains unwanted phrases"""
    unwanted_phrases = [
        "for more information, visit www.fema.gov",
        "visit www.fema.gov",
        "www.fema.gov",
        "fema.gov",
        "for more information",
        "please visit"
    ]
    return any(phrase in text.lower() for phrase in unwanted_phrases)

def is_repeating_phrase(text: str) -> bool:
    """Check if transcript contains repeating sentences"""
    if not text or len(text) < 10:
        return False

    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) < 3:
        return False

    # Check for repeating sentences
    first_sentence = sentences[2].lower()
    repetition_count = sum(1 for s in sentences if s.lower() == first_sentence)
    
    return repetition_count >= 3


async def process_audio(audio_data: bytes) -> str:
    """Convert raw PCM bytes to WAV and transcribe via Whisper"""
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data)
        
        wav_buffer.seek(0)

        resp = await async_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("audio.wav", wav_buffer, "audio/wav"),
            language="en",
            temperature=0.3,
            prompt="This is a conversation about booking a hospital appointment."
        )
        return resp.text

    except Exception as e:
      
        raise

async def execute_tool_call(function_name, arguments):
    """Execute tool functions and return results"""
    try:

        if function_name == "submit_booking":
            booking_data = {
                "name": arguments.get("name"),
                "phone": arguments.get("phone"),
                "date": arguments.get("date"),
                "time": arguments.get("time")
            }
            
           
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:6000/api/bookings",
                    json=booking_data,
                    timeout=10.0
                )
                
                
                if response.status_code == 200:
                   
                    return {"status": "success", "message": "Booking confirmed!"}
                else:
                    
                    return {"status": "error", "message": "Failed to submit booking"}
        
        return {"status": "success"}
            
    except Exception as e:
       
        return {"status": "error", "message": str(e)}

# Add this function before generate_response_with_history
async def generate_fallback_response() -> tuple[bytes, str, list]:
    """Generate a fallback response when an error occurs"""
    try:
        fallback_text = "I apologize, but I encountered an error. Could you please repeat that?"
        
        # Generate speech for fallback message
        speech_resp = await async_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=fallback_text,
            response_format="mp3"
        )
        
        return speech_resp.content, fallback_text, []
    except Exception as e:
       
        # Return empty audio, error message, and empty tool messages if even fallback fails
        return b"", "Sorry, I'm having technical difficulties.", []


async def generate_response_with_history(text: str, conversation_history: list, stored_info: dict = None) -> tuple[bytes, str, list]:
    try:
        # Extract booking information
        booking_info = {}
        
        # Name extraction
        if "name" in text.lower() or "i am" in text.lower():
            name_match = text.lower().split("name is ")[-1].split(".")[0] if "name is" in text.lower() else \
                        text.lower().split("i am ")[-1].split(".")[0]
            if name_match:
                booking_info["name"] = name_match.strip().title()

        # Update stored info
        if stored_info is not None and booking_info:
            
            stored_info.update(booking_info)
            

        # Generate response
        chat_resp = await async_client.chat.completions.create(
            model=GPT_MODEL,
            messages=conversation_history + [
                {"role": "system", "content": f"Current booking info: {json.dumps(stored_info)}"},
                {"role": "user", "content": text}
            ],
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=150
        )

        response_text = chat_resp.choices[0].message.content
        tool_messages = []

        # Ensure we have valid response text
        if not response_text or not isinstance(response_text, str):
            response_text = "I apologize, could you please repeat that?"
        
        # Generate speech
        try:
            speech_resp = await async_client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=response_text.strip(),
                response_format="mp3"
            )
            return speech_resp.content, response_text, tool_messages
        except Exception as tts_error:

            return b"", response_text, tool_messages

    except Exception as e:

        return await generate_fallback_response()

async def voice_websocket_endpoint(ws: WebSocket):
    await ws.accept()
   

    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    stored_info: dict[str, str] = {}
    audio_buffer = bytearray()
    vad_processor = EnhancedVADProcessor(sample_rate=16000, min_duration=0.8, min_rms=350.0)
    mode         = "push-to-talk"
    is_connected = True
    noise_count = 0

    async def process_response(transcript: str):
        """
        Append user transcript, send it to GPT, and stream the TTS back.
        """
        nonlocal conversation_history, stored_info, is_connected,noise_count
        words = transcript.split()
        if not transcript or len(transcript) < 4 or len(words) < 1:
            noise_count += 1

            if noise_count >= 3:
                conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                noise_count = 0
                try:
                    await ws.send_json({
                        "type": "chat_noise",
                        "content": "I notice there's some background noise. I'll switch to push-to-talk mode to help us communicate more clearly."
                    })
                except Exception:
                
                    return
            return 

        if is_unwanted_transcript(transcript) or is_repeating_phrase(transcript):
            noise_count+=1
            return

        noise_count = 0
        
        if not is_connected:
            return

        try:
            conversation_history.append({"role": "user", "content": transcript})
            await ws.send_json({"type": "chat_response", "transcript": transcript})

            audio_content, response_text, tool_messages = await generate_response_with_history(
                transcript, conversation_history, stored_info
            )

            for tool_msg in tool_messages:
                if not is_connected:
                    return
                if tool_msg.get("name") == "set_memory":
                    stored_info.update(tool_msg["arguments"])
                await ws.send_json({"type": "tool_call", "data": tool_msg})

            if not is_connected:
                return

            conversation_history.append({"role": "assistant", "content": response_text})
            await ws.send_json({"type": "chat_response", "content": response_text})
            if audio_content:
                await ws.send_bytes(audio_content)

        except WebSocketDisconnect:
            is_connected = False
        except Exception as e:
          
            if is_connected:
                try:
                    await ws.send_json({"type": "error", "content": "Error generating response"})
                except:
                    is_connected = False

    try:
        while is_connected:
            try:
                message = await ws.receive()

                if "text" in message:
                    payload = json.loads(message["text"])

                    if payload.get("event") == "initial_message":
                        greeting = "Hello! How can I assist you today?"
                        try:
                            speech_resp = await async_client.audio.speech.create(
                                model=TTS_MODEL,
                                voice=TTS_VOICE,
                                input=greeting,
                                response_format="mp3",
                            )
                            if is_connected:
                                conversation_history.append({"role": "assistant", "content": greeting})
                                await ws.send_json({"type": "chat_response", "content": greeting})
                                await ws.send_bytes(speech_resp.content)
                        except Exception as e:

                            if is_connected:
                                await ws.send_json({"type": "error", "content": "Error generating speech"})

                        mode = payload.get("mode", "push-to-talk")
                  

                    elif payload.get("event") == "set_mode":
                        mode = payload.get("mode", "push-to-talk")
                        audio_buffer.clear()
       
                    elif payload.get("event") == "end_of_turn" and mode == "push-to-talk":
                        if len(audio_buffer) >= 4096:
                            try:
                                transcript = await process_audio(bytes(audio_buffer))
                                audio_buffer.clear()
                                if transcript and transcript.strip():
                                    await process_response(transcript)
                                else:
                                    if is_connected:
                                        await ws.send_json({
                                            "type": "chat_response",
                                            "content": "I couldn't hear that clearly. Please speak again."
                                        })
                            except Exception as e:
                            
                                if is_connected:
                                    await ws.send_json({"type": "error", "content": "Error processing audio"})
                        else:
                            if is_connected:
                                await ws.send_json({
                                    "type": "chat_response",
                                    "content": "I couldn't hear anything. Please speak again."
                                })

                elif "bytes" in message:
                    chunk = message["bytes"]
                    if not is_connected:
                        break

                    if mode == "real-time":
                        vad_result = vad_processor.process(chunk)
                        if vad_result:

                            try:
                                transcript = await process_audio(vad_result)
                                await process_response(transcript)        
                            except Exception as e:
                         
                                if is_connected:
                                    await ws.send_json({"type": "error", "content": "Error processing audio"})
                    else:
                        audio_buffer.extend(chunk)

            except WebSocketDisconnect:
              
                is_connected = False
            except Exception as e:
               
                if "disconnect" in str(e).lower():
                    is_connected = False

    finally:
        is_connected = False
    