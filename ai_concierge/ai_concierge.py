import boto3
import json
from botocore.exceptions import ClientError
from bedrock_agentcore.memory import MemoryClient
from strands import Agent
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent

REGION = 'us-east-1'
MEMORY_NAME = "AIConciergeMemory"
ACTOR_ID = "concierge_user"

client = MemoryClient(region_name=REGION)

MEMORY_ID = None
try:
    memory = client.create_memory_and_wait(
        name=MEMORY_NAME,
        strategies=[],
        description="Short-term memory for AI concierge",
        event_expiry_days=15,
    )
    MEMORY_ID = memory['id']
    print(f"Created memory: {MEMORY_ID}")
except ClientError as e:
    if 'already exists' in str(e):
        memories = client.list_memories()
        MEMORY_ID = next((m['id'] for m in memories if m['id'].startswith(MEMORY_NAME)), None)
        if MEMORY_ID:
            print(f"Using existing memory: {MEMORY_ID}")
        else:
            raise ValueError(f"Memory with name {MEMORY_NAME} not found in existing memories")
    else:
        raise e
if not MEMORY_ID:
    raise ValueError("Failed to initialize memory ID")

class MemoryHookProvider(HookProvider):
    def __init__(self, memory_client: MemoryClient, memory_id: str, actor_id: str, session_id: str):
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
    
    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load recent conversation history when agent starts"""
        try:
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=10
            )
            
            if recent_turns:
                context_messages = []
                for turn in recent_turns:
                    for message in turn:
                        role = message['role']
                        content = message['content']['text']
                        context_messages.append(f"{role}: {content}")
                
                context = "\n".join(context_messages)
                event.agent.system_prompt += f"\n\nRecent conversation:\n{context}"
                print(f"Loaded {len(recent_turns)} conversation turns for session {self.session_id}")
                
        except Exception as e:
            print(f"Memory load error: {e}")
    
    def on_message_added(self, event: MessageAddedEvent):
        """Store messages in memory"""
        messages = event.agent.messages
        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(messages[-1]["content"][0]["text"], messages[-1]["role"])]
            )
            print(f"Saved new message for session {self.session_id}")
        except Exception as e:
            print(f"Memory save error: {e}")
    
    def register_hooks(self, registry: HookRegistry):
        # Register memory hooks
        registry.add_callback(MessageAddedEvent, self.on_message_added)
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)

def ai_concierge(session_id: str, question: str):
    system_prompt = """
    You are an AI concierge specializing in providing business-related information. Your role is to answer questions directly related to businesses, including but not limited to:
    Operating hours, Locations and addresses, Products or services offered, Customer reviews and ratings, Contact details (e.g., phone numbers, emails), Business policies (returns, shipping, etc.)

    If a question references previous interactions, you may use relevant information from past conversations to provide a more tailored and informed response.

    If the question is unrelated to businesses or off-topic, politely respond with: "I'm sorry, I can only answer questions related to businesses."
    Keep your responses clear, concise, factual, and relevant. Avoid opinions, personal suggestions, or anything unrelated to business information.
    """
    
    agent = Agent(
        name="AIConcierge",
        system_prompt=system_prompt,
        hooks=[MemoryHookProvider(client, MEMORY_ID, ACTOR_ID, session_id)],
    )
    try:
        result = agent(question)
        answer = result.message['content'][0]['text']
    except Exception as e:
        print(f"Error generating response: {e}")
        answer = "I'm sorry, there was an error processing your request."
    
    return answer
