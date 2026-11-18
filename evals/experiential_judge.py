
import os
from dotenv import load_dotenv
from openai import OpenAI
from deepeval.models import DeepEvalBaseLLM

load_dotenv()


class ExperientialLabsJudge(DeepEvalBaseLLM):

    def __init__(self):
        self.model_name = "gpt-4o-mini"

        api_key = os.getenv("EXPLABS_API_KEY")

        if not api_key:
            raise ValueError("EXPLABS_API_KEY not found in .env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.experientiallabs.ai/v1"
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name
