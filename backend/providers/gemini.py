# FILE: backend/providers/gemini.py
"""
Gemini (Google) provider adapter
"""
import logging
from typing import Dict, Any
# import google.generativeai as genai
from google import genai

logger = logging.getLogger(__name__)


# class GeminiProvider:
#     """Gemini provider (Google)"""
    
#     def __init__(self, api_key: str, model: str):
#         genai.configure(api_key=api_key)
#         self.model_name = model
#         self.model = genai.GenerativeModel(model)
#         logger.info(f"Gemini provider: model={model}")
    
#     def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
#         """Generate text using Gemini"""
#         generation_config = {
#             "temperature": temperature,
#             "max_output_tokens": max_tokens
#         }
        
#         response = self.model.generate_content(
#             prompt,
#             generation_config=generation_config
#         )
        
#         text = response.text
        
#         return {
#             "text": text,
#             "model": self.model_name,
#             "usage": {}
#         }

class GeminiProvider:
    """Gemini provider (Google)"""
    
    def __init__(self, api_key: str, model: str):
        # FIX 1: Replace genai.configure() with Client initialization.
        self.client = genai.Client(api_key=api_key) 
        self.model_name = model
        # Remove the old self.model assignment (self.model = genai.GenerativeModel(model))
        # We will use self.client in the generate method instead.
        logger.info(f"Gemini provider: model={model}")
    
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Generate text using Gemini"""
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }
        
        # FIX 2: Call generate_content via the client, passing the model name.
        response = self.client.models.generate_content( 
            model=self.model_name, # Specify the model here
            contents=prompt, # The 'prompt' is passed as 'contents' in the new SDK
            config=generation_config
        )
        
        text = response.text
        
        return {
            "text": text,
            "model": self.model_name,
            "usage": {}
        }
