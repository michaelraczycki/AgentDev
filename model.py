import os
import time
import uuid
import json
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

model = init_chat_model(
    "qwen2.5-coder",
    model_provider="ollama",
    base_url="http://localhost:11434",  # default Ollama URL
)
class ModelConfig:
    def __init__(self, model_name: str, model_provider: str = "ollama", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.model_provider = model_provider
        self.base_url = base_url

class MessageStore:
    """Manages raw message storage. No model or session logic."""
    def __init__(self, session_id: str, user_id: str):
        self.session_id = session_id
        self.user_id = user_id
        self._messages: list[BaseMessage] = []

    def clear(self):
        self._messages = []
    
    def add_user_message(self, content: str):
        self._messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str, mode: str = ""):
        self._messages.append(AIMessage(content=content, response_metadata={"mode": mode}))
    
    def get_display_history(self) -> list:
        display_history = []
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                display_history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                mode = msg.response_metadata.get("mode", "")
                display_history.append({"role": "assistant", "content": msg.content, "mode": mode})
        return display_history
    
    def set_system_prompt(self, prompt: str):
        if self._messages and isinstance(self._messages[0], SystemMessage):
            self._messages[0] = SystemMessage(content=prompt)
        else:
            self._messages.insert(0, SystemMessage(content=prompt))
    
    def set_compressed_history(self, compressed_content: str):
        self._messages.insert(1,SystemMessage(content=f"[Compressed history]: {compressed_content}"))

    def get_messages(self) -> list:
        return self._messages.copy()

    def pop_last(self):
        if len(self._messages) > 1:
            self._messages.pop()
    
    def count_tokens(self) -> int:
        # Placeholder token counting logic. In production, use model-specific tokenization.
        return sum(len(str(msg.content)) for msg in self._messages) // 4

    def _serialize_message(self, msg: BaseMessage) -> dict:
        base = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": time.time(),
        }
        if isinstance(msg, SystemMessage):
            return {**base, "role": "system", "content": msg.content, "message_type": "SystemMessage"}
        elif isinstance(msg, HumanMessage):
            return {**base, "role": "user", "content": msg.content, "message_type": "HumanMessage"}
        else:
            return {**base, "role": "assistant", "content": msg.content, "message_type": "AIMessage", "mode": msg.response_metadata.get("mode", "")}
        
    def flush_to_json(self):
        path = Path(f"storage/{self.user_id}/{self.session_id}_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        flushed = self._messages[:-4]

        with open(path, "w") as f:
            json.dump([self._serialize_message(msg) for msg in flushed], f, indent=2)
        
        self._messages = self._messages[-4:]
        return flushed


class ChatSession:
    def __init__(self, system_prompt: str, user_id: str = 'default'):
        self.session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.user_id = user_id
        self.system_prompt = system_prompt
        self.store = MessageStore(session_id=self.session_id, user_id=self.user_id)

    def clear(self):
        self.store.clear()
        self.update_system_prompt(self.system_prompt)

    def pop_last(self):
        self.store.pop_last()

    def get_messages(self) -> list:
        return self.store.get_messages()
    
    def get_display_history(self) -> list:
        return self.store.get_display_history()

    def update_system_prompt(self, new_prompt: str):
        self.system_prompt = new_prompt
        self.store.set_system_prompt(new_prompt)


class ChatBot:
    def __init__(self, model_config: ModelConfig, system_prompt: str):
        self.model = init_chat_model(
            model_config.model_name,
            model_provider=model_config.model_provider,
            base_url=model_config.base_url,
        )
        self.session: ChatSession = ChatSession(system_prompt=system_prompt if system_prompt else "You are a helpful assistant.")
        
    def try_to_compress(self):
        if self.session.store.count_tokens() > 10_000:
            messages = self.session.store.flush_to_json()
            self._compress(messages)

    def chat(self, user_input: str)-> str:
        self.session.store.add_user_message(user_input)
        self.try_to_compress()
        
        try:
            response = self.model.invoke(self.session.store.get_messages())
            self.session.store.add_ai_message(str(response.content), mode="invoke")
            return str(response.content)
        except Exception as e:
            return "Sorry, there was an error processing your request."
        
    def chat_stream(self, user_input: str):
        self.session.store.add_user_message(user_input)
        self.try_to_compress()
        try:
            full_response = ""
            for chunk in self.model.stream(self.session.store.get_messages()):
                full_response += str(chunk.content)
                yield full_response
            self.session.store.add_ai_message(full_response, mode="stream")
        except Exception as e:
            yield "Sorry, there was an error processing your request."

    def _compress(self, messages: list[BaseMessage]):
        # Placeholder compression logic. In production, use a more sophisticated approach.
        if len(messages) <= 4:
            return  # Not enough messages to compress
        messages_to_compress = " ".join(str(msg.content) for msg in messages)
        prompt = (
            "Given the messages, prepare a summary that best reflects the core issue "
            "of the conversation, or if there is more than 1 topic present, "
            f"list of short summaries of those topics:\n{messages_to_compress}"
        )
        compressed_content = self.model.invoke([SystemMessage(content=prompt)])
        self.session.store.set_system_prompt(self.session.system_prompt)
        self.session.store.set_compressed_history(compressed_content.content)

    

system_prompt = input("Enter system prompt (or press Enter for default): ")
if system_prompt.strip() == "":
    system_prompt = "You are a helpful assistant."
convo = ChatBot(ModelConfig(model_name="llama3.2"), system_prompt=system_prompt)
while True:
    question = input("ask a question:")
    if question.lower() in ["exit", "quit"]:
        break
    response = convo.chat(question)
    print("Response:", response)