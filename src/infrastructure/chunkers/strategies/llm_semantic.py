import cocoindex
import re
from tqdm import tqdm
import tiktoken

from typing import List

from src.infrastructure.chunkers.strategies.recursive import recursive_chunk_text
from src.infrastructure.clients.openai_client import OpenAIClient
from src.domain.models import TextChunk

def openai_token_count(text: str) -> int:
    """Count tokens using OpenAI's tokenizer (tiktoken)"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base") # OpenAI 모델은 보통 cl100k_base 인코딩 사용
        return len(encoding.encode(text))
    except Exception:
        raise ValueError("Failed to count tokens")

@cocoindex.op.function()
def llm_semantic_chunk(
    text: str,
    organization: str = "openai",
    api_key: str = None,
    model_name: str = None,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
) -> List[TextChunk]:    
    
    if organization == "openai":
        if model_name is None:
            model_name = "gpt-4o-mini"
        client = OpenAIClient(model_name, api_key)
    else:
        raise ValueError(f"Unsupported organization: {organization}")
    
    chunks = recursive_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_indices = []
    current_chunk = 0

    def get_prompt(chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk)+"." + (f"\n\\The previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages
    
    with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
        while True:
            if current_chunk >= len(chunks) - 4:
                break

            token_count = 0
            chunked_input = ''

            # 원본 로직: 각 청크의 토큰만 계산하고 chunked_input에 추가
            for i in range(current_chunk, len(chunks)):
                token_count += openai_token_count(chunks[i])
                chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
                if token_count > 800:  # https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/llm_semantic_chunker.py 참고
                    break

            messages = get_prompt(chunked_input, current_chunk)
            while True:
                result_string = client.create_message(messages[0]['content'], messages[1:], max_tokens=200, temperature=0.2)
                split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line][0]
                numbers = list(map(int, re.findall(r'\d+', split_after_line)))

                # Check if the numbers are in ascending order and are equal to or larger than current_chunk
                if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)): # 둘 다 참이면 조건 성립
                    break
                else:
                    messages = get_prompt(chunked_input, current_chunk, numbers)
                    print("Response: ", result_string)
                    print("Invalid response. Please try again.")

            split_indices.extend(numbers)
            current_chunk = numbers[-1]

            if len(numbers) == 0:
                break

            pbar.update(current_chunk - pbar.n)

    pbar.close()

    chunks_to_split_after = [i - 1 for i in split_indices]
    
    docs = []
    current_chunk_text = ''
    for i, chunk in enumerate(chunks):
        current_chunk_text += chunk + ' '
        if i in chunks_to_split_after:
            docs.append(current_chunk_text.strip())
            current_chunk_text = ''
    if current_chunk_text:
        docs.append(current_chunk_text.strip())

    return [TextChunk(text=doc) for doc in docs]