import os
import base64
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import time
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv()

# Retrieve credentials from .env file or environment
endpoint = os.getenv("gpt_endpoint")
deployment = os.getenv("gpt_deployment")
api_version = os.getenv("gpt_api_version")


def _resolve_token_scope(azure_endpoint: str) -> str:
    """Pick the correct AAD audience based on endpoint family."""
    if "services.ai.azure.com" in azure_endpoint:
        return "https://ai.azure.com/.default"
    return "https://cognitiveservices.azure.com/.default"


def _is_foundry_endpoint(azure_endpoint: str) -> bool:
    return "services.ai.azure.com" in azure_endpoint


def _to_foundry_base_url(azure_endpoint: str) -> str:
    # Accept either the host root or a full /openai/v1 URL from env.
    if azure_endpoint.endswith("/openai/v1"):
        return azure_endpoint
    return azure_endpoint.rstrip("/") + "/openai/v1"


if not endpoint:
    raise ValueError("gpt_endpoint is required")
if not deployment:
    raise ValueError("gpt_deployment is required")
if not api_version and not _is_foundry_endpoint(endpoint):
    raise ValueError("gpt_api_version is required")

# Initialize Azure OpenAI client for GPT model using managed identity
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, _resolve_token_scope(endpoint))
if _is_foundry_endpoint(endpoint):
    client = OpenAI(
        base_url=_to_foundry_base_url(endpoint),
        api_key=token_provider,
    )
else:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )

def generate_response(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    # Prepare the full chat prompt with system and user messages
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                        Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                        If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                        Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
                    """
                }
            ]
        },
        {"role": "user", "content": text_input}
    ]

    # Call Azure OpenAI chat API. Prefer max_completion_tokens for newer models,
    # then fall back to max_tokens for older model families.
    request_kwargs = {
        "model": deployment,
        "messages": chat_prompt,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "stream": False,
    }

    try:
        completion = client.chat.completions.create(
            max_completion_tokens=4000,
            **request_kwargs,
        )
    except Exception as e:
        error_text = str(e)
        if "max_completion_tokens" in error_text and "unsupported" in error_text.lower():
            completion = client.chat.completions.create(
                max_tokens=4000,
                **request_kwargs,
            )
        else:
            raise
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")
    # Return response content
    return completion.choices[0].message.content
