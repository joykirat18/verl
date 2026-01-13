import pandas as pd
import os

import os
import re
import time
from openai import AzureOpenAI, OpenAI
from openai import RateLimitError, APIConnectionError, APIError, APITimeoutError
from typing import List, Dict, Optional


test_data = "/nas-ssd2/joykirat/code/state-representation/verl/scripts/data/blocksworld_state_action/train.parquet"
model = "o4-mini"
test_data = pd.read_parquet(test_data)

# breakpoint()

messages = []

for i in range(len(test_data)):
    message = test_data['prompt'][i]
    messages.append(message)

def client_azure(base_url, api_key):

    if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    api_version = "2024-12-01-preview"
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=base_url,
        api_key=api_key,
    )

    return client
    
def client_openai(base_url, api_key):

    if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    client = OpenAI(api_key=api_key, base_url=base_url)

    return client


def query_model(
    message, 
    model="gpt-4", 
    api_key=None, 
    base_url=None, 
    return_raw=False,
    backend="azure",  # "azure", "openai", or "vllm"
    vllm_base_url=None,
    max_retries=3,
    initial_backoff=1.0,
    max_backoff=60.0,
    backoff_multiplier=2.0
):
    """
    Query model API (Azure OpenAI, OpenAI, or vLLM) with a message.
    
    Returns:
        If return_raw=False: Model's extracted answer as a string (lowercased and stripped)
        If return_raw=True: Dict with keys: 'raw_response', 'extracted_answer', 'final_prediction'
    """
    # Handle API-based models
    # Get API key from parameter, environment variable, or raise error
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    endpoint = "https://joykirat-api.cognitiveservices.azure.com/"

    if model == "gpt-4o-mini":
        max_tokens = 16384
        client = client_azure(endpoint, api_key)
    elif model == "gpt-4o":
        max_tokens = 16384
        client = client_azure(endpoint, api_key)
    elif model == "o4-mini":
        max_tokens = 32000
        client = client_azure(endpoint, api_key)
    elif model == "DeepSeek-R1":
        max_tokens = 32000
        client = client_openai("https://joykirat-api.services.ai.azure.com/openai/v1/", api_key)
    elif model == "gpt-oss-120b":
        max_tokens = 32000
        client = client_openai("https://joyki-mjn6s9tj-eastus2.services.ai.azure.com/openai/v1/", api_key)
    else:
        raise ValueError(f"Model {model} not supported")

    # Retry mechanism with exponential backoff
    last_exception = None
    backoff_time = initial_backoff
    
    for attempt in range(3):
        try:
            if model == "o4-mini":
                response = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=1.0,
                max_completion_tokens=max_tokens,
            )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=1.0,
                    max_completion_tokens=max_tokens,
                )
            content = response.choices[0].message.content
            if content is None:
                print(f"API response has no content for model {model}. This may indicate an error or empty response.")
                content = "ERROR"
            raw_response = content.strip()
            
            # Extract answer from <answer>...</answer> tags
            return raw_response
                
        except (RateLimitError, APIConnectionError, APITimeoutError) as e:
            # Retryable errors: rate limits, connection issues, timeouts
            last_exception = e
            if attempt < max_retries:
                wait_time = min(backoff_time, max_backoff)
                print(f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}. "
                      f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                backoff_time *= backoff_multiplier
            else:
                print(f"Max retries ({max_retries}) exceeded for {type(e).__name__}")
                
        except APIError as e:
            # Check if it's a retryable API error (e.g., 500, 502, 503)
            status_code = getattr(e, 'status_code', None)
            if status_code and status_code in [500, 502, 503, 504] and attempt < max_retries:
                last_exception = e
                wait_time = min(backoff_time, max_backoff)
                print(f"Retryable API error {status_code} (attempt {attempt + 1}/{max_retries + 1}). "
                      f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                backoff_time *= backoff_multiplier
            else:
                # Non-retryable API error or max retries reached
                raise RuntimeError(f"Error calling API: {e}") from e
                
        except Exception as e:
            # Non-retryable errors (e.g., authentication, invalid request)
            raise RuntimeError(f"Error calling API: {e}") from e
    
    # If we exhausted all retries, raise the last exception
    if last_exception:
        raise RuntimeError(f"Error calling API after {max_retries} retries: {last_exception}") from last_exception

import json
final_responses = []
if not os.path.exists(f'{model}_responses_with_state_action_train.json'):
    final_responses = []
else:
    with open(f'{model}_responses_with_state_action_train.json', 'r') as f:
        final_responses = json.load(f)

start_index = len(final_responses)

from tqdm import tqdm

    
for i in tqdm(range(start_index, len(messages))):
    message = messages[i]

    response = query_model(message, model)

    final_responses.append({'question': message[0]['content'], 'response': response})

    import json
    with open(f'{model}_responses_with_state_action_train.json', 'w') as f:
        json.dump(final_responses, f)



    


