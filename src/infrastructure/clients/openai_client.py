import backoff
from openai import OpenAI

class OpenAIClient:
    def __init__(self, model_name, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        gpt_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=gpt_messages,
            temperature=temperature
        )
        return completion.choices[0].message.content