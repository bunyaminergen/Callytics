# Standard library imports
import os
import json
import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Any, Annotated

# Related third-party imports
import yaml
import torch
import openai
from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()


class LanguageModel(ABC):
    """
    Abstract base class for language models.

    This class provides a common interface for language models with methods
    to generate text and unload resources.

    Parameters
    ----------
    config : dict
        Configuration for the language model.
    """

    def __init__(self, config: Annotated[dict, "Configuration for the language model"]):
        self.config = config

    @abstractmethod
    def generate(
            self,
            messages: Annotated[list, "List of message dictionaries"],
            **kwargs: Annotated[Any, "Additional keyword arguments"]
    ) -> Annotated[str, "Generated text"]:
        """
        Generate text based on the given input messages.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            Generated text output.
        """
        pass

    def unload(self) -> Annotated[None, "Unload resources used by the language model"]:
        """
        Unload resources used by the language model.
        """
        pass


class LLaMAModel(LanguageModel):
    """
    LLaMA language model implementation using Hugging Face Transformers.

    Parameters
    ----------
    config : dict
        Configuration for the LLaMA model.
    """

    def __init__(self, config: Annotated[dict, "Configuration for the LLaMA model"]):
        super().__init__(config)
        model_name = config['model_name']
        compute_type = config.get('compute_type')
        torch.cuda.empty_cache()

        print(f"Loading LLaMA model: {model_name}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU not available, using CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and compute_type == "float16" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

    def generate(
            self,
            messages: Annotated[list, "List of message dictionaries"],
            max_new_tokens: Annotated[int, "Maximum number of new tokens to generate"] = 10000,
            truncation: Annotated[bool, "Whether to truncate the input"] = True,
            batch_size: Annotated[int, "Batch size for generation"] = 1,
            pad_token_id: Annotated[Optional[int], "Padding token ID"] = None
    ) -> Annotated[str, "Generated text"]:
        """
        Generate text based on input messages using the LLaMA model.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'.
        max_new_tokens : int, optional
            Maximum number of tokens to generate. Default is 10000.
        truncation : bool, optional
            Whether to truncate the input. Default is True.
        batch_size : int, optional
            Batch size for generation. Default is 1.
        pad_token_id : int, optional
            Padding token ID. Defaults to the tokenizer's EOS token ID.

        Returns
        -------
        str
            Generated text.
        """
        prompt = self._format_messages_llama(messages)
        output = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            truncation=truncation,
            batch_size=batch_size,
            pad_token_id=pad_token_id if pad_token_id is not None else self.tokenizer.eos_token_id
        )
        return output[0]['generated_text']

    @staticmethod
    def _format_messages_llama(messages: Annotated[list, "List of message dictionaries"]) -> Annotated[
        str, "Formatted prompt"]:
        """
        Format messages into a single prompt for LLaMA.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'.

        Returns
        -------
        str
            Formatted prompt.
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

    def unload(self) -> Annotated[None, "Unload the LLaMA model and release resources"]:
        """
        Unload the LLaMA model and release resources.
        """
        del self.pipe
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        print(f"LLaMA model '{self.config['model_name']}' unloaded.")


class OpenAIModel(LanguageModel):
    """
    OpenAI GPT model integration.

    Parameters
    ----------
    config : dict
        Configuration for the OpenAI model.
    """

    def __init__(self, config: Annotated[dict, "Configuration for the OpenAI model"]):
        super().__init__(config)
        openai_api_key = config.get('openai_api_key')
        if not openai_api_key:
            raise ValueError("OpenAI API key must be provided.")
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = config.get('model_name', 'gpt-4')

    def generate(
            self,
            messages: Annotated[list, "List of message dictionaries"],
            max_length: Annotated[int, "Maximum number of tokens for the output"] = 10000,
            return_as_json: bool = False,
            **kwargs: Annotated[Any, "Additional keyword arguments"]
    ) -> Annotated[str, "Generated text"]:
        """
        Generate text using OpenAI's API.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'.
        max_length : int, optional
            Maximum number of tokens for the output. Default is 10000.
        return_as_json : bool, optional
            If True, response_format={"type": "json_object"} parametresi eklenir ve dönen içerik
            json.loads ile dict'e dönüştürülür. Varsayılan False'dur.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str or dict
            Generated text as a string if return_as_json=False.
            If return_as_json=True and the response is in valid JSON format,
            returns a dict.
        """

        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_length,
            "temperature": kwargs.get('temperature', 0.7)
        }

        if return_as_json is True:
            create_kwargs["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**create_kwargs)
        response_text = completion.choices[0].message.content

        if return_as_json:
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return response_text

        return response_text

    def unload(self) -> Annotated[None, "Placeholder for OpenAI model unload (no local resources to release)"]:
        """
        Placeholder for OpenAI model unload (no local resources to release).
        """
        print(f"OpenAI model '{self.model_name}' unloaded.")


class AzureOpenAIModel(LanguageModel):
    """
    Azure OpenAI model integration.

    Parameters
    ----------
    config : dict
        Configuration for the Azure OpenAI model.
    """

    def __init__(self, config: Annotated[dict, "Configuration for the Azure OpenAI model"]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'gpt-4o')
        self.api_key = config.get('azure_openai_api_key')
        self.api_base = config.get('azure_openai_api_base')
        self.api_version = config.get('azure_openai_api_version')

        if not all([self.api_key, self.api_base, self.api_version]):
            raise ValueError("Azure OpenAI API key, base, and version must be provided.")

        openai.api_type = "azure"
        openai.api_base = self.api_base
        openai.api_version = self.api_version
        openai.api_key = self.api_key

    def generate(
            self,
            messages: Annotated[list, "List of message dictionaries"],
            max_length: Annotated[int, "Maximum number of tokens for the output"] = 10000,
            **kwargs: Annotated[Any, "Additional keyword arguments"]
    ) -> Annotated[str, "Generated text"]:
        """
        Generate text using Azure OpenAI's API.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'.
        max_length : int, optional
            Maximum number of tokens for the output. Default is 10000.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str
            Generated text.
        """
        response = openai.ChatCompletion.create(
            deployment_id=self.model_name,
            messages=messages,
            max_tokens=max_length,
            temperature=kwargs.get('temperature', 0.7)
        )
        return response.choices[0].message['content']

    def unload(self) -> Annotated[None, "Placeholder for Azure OpenAI model unload (no local resources to release)"]:
        """
        Placeholder for Azure OpenAI model unload (no local resources to release).
        """
        print(f"Azure OpenAI model '{self.model_name}' unloaded.")


class ModelRegistry:
    """
    Registry to manage language model class registrations.

    This class allows dynamic registration and retrieval of model classes.
    """
    _registry = {}

    @classmethod
    def register(
            cls,
            model_id: Annotated[str, "Unique identifier for the model"],
            model_class: Annotated[type, "The class to register"]
    ) -> Annotated[None, "Registration completed"]:
        """
        Register a model class with the registry.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model class.
        model_class : type
            The class to register.
        """
        cls._registry[model_id.lower()] = model_class

    @classmethod
    def get_model_class(cls, model_id: Annotated[str, "Unique identifier for the model"]) -> Annotated[
        type, "Model class"]:
        """
        Retrieve a model class by its unique identifier.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model class.

        Returns
        -------
        type
            The model class corresponding to the identifier.

        Raises
        ------
        ValueError
            If the model ID is not registered.
        """
        model_class = cls._registry.get(model_id.lower())
        if not model_class:
            raise ValueError(f"No class found for model ID '{model_id}'.")
        return model_class


class ModelFactory:
    """
    Factory to create language model instances.

    This class uses the `ModelRegistry` to create instances of registered model classes.
    """

    @staticmethod
    def create_model(
            model_id: Annotated[str, "Unique identifier for the model"],
            config: Annotated[dict, "Configuration for the model"]
    ) -> Annotated[LanguageModel, "Instance of the language model"]:
        """
        Create a language model instance based on its unique identifier.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        config : dict
            Configuration for the model.

        Returns
        -------
        LanguageModel
            An instance of the language model.
        """
        model_class = ModelRegistry.get_model_class(model_id)
        return model_class(config)


class LanguageModelManager:
    """
    Manages multiple language models with caching and async support.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    cache_size : int, optional
        Maximum number of models to cache. Default is 10.
    """

    def __init__(
            self,
            config_path: Annotated[str, "Path to the YAML configuration file"],
            cache_size: Annotated[int, "Maximum number of models to cache"] = 10
    ):
        self.config_path = config_path
        self.cache_size = cache_size
        self.models = OrderedDict()
        self.full_config = self._load_full_config(config_path)
        self.runtime_config = self.full_config.get('runtime', {})
        self.models_config = self.full_config.get('models', {})
        self.lock = asyncio.Lock()

    @staticmethod
    def _load_full_config(config_path: Annotated[str, "Path to the YAML configuration file"]) -> Annotated[
        dict, "Parsed configuration"]:
        """
        Load and parse the YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML file.

        Returns
        -------
        dict
            Parsed configuration.
        """
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for model_id, model_config in config.get('models', {}).items():
            for key, value in model_config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    model_config[key] = os.getenv(env_var, "")
        return config

    async def get_model(
            self,
            model_id: Annotated[str, "Unique identifier for the model"]
    ) -> Annotated[LanguageModel, "Instance of the language model"]:
        """
        Retrieve a language model instance from the cache or create a new one.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        LanguageModel
            An instance of the language model.

        Raises
        ------
        ValueError
            If the model ID is not found in the configuration.
        """
        async with self.lock:
            torch.cuda.empty_cache()
            if model_id in self.models:
                self.models.move_to_end(model_id)
                return self.models[model_id]
            else:
                config = self.models_config.get(model_id)
                if not config:
                    raise ValueError(f"Model ID '{model_id}' not found in configuration.")
                config['compute_type'] = self.runtime_config.get('compute_type', 'float16')
                model = ModelFactory.create_model(model_id, config)
                self.models[model_id] = model
                if len(self.models) > self.cache_size:
                    oldest_model_id, oldest_model = self.models.popitem(last=False)
                    oldest_model.unload()
                return model

    async def generate(
            self,
            model_id: Annotated[str, "Unique identifier for the model"],
            messages: Annotated[list, "List of message dictionaries"],
            **kwargs: Annotated[Any, "Additional keyword arguments"]
    ) -> Annotated[Optional[str], "Generated text or None if an error occurs"]:
        """
        Generate text using a specific language model.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        messages : list
            List of message dictionaries with 'role' and 'content'.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        str or None
            Generated text or None if an error occurs.
        """
        try:
            model = await self.get_model(model_id)
            return model.generate(messages, **kwargs)
        except Exception as e:
            print(f"Error with model ({model_id}): {e}")
            return None

    def unload_all(self) -> Annotated[None, "Unload all cached models and release resources"]:
        """
        Unload all cached models and release resources.
        """
        for model in self.models.values():
            model.unload()
        self.models.clear()
        print("All models have been unloaded.")


if __name__ == "__main__":
    # noinspection PyMissingOrEmptyDocstring
    async def main():
        config_path = 'config/config.yaml'

        manager = LanguageModelManager(config_path=config_path, cache_size=11)

        llama_model_id = "llama"
        llama_messages = [
            {"role": "system", "content": "You are a pirate. Answer accordingly!"},
            {"role": "user", "content": "Who are you?"}
        ]
        llama_output = await manager.generate(model_id=llama_model_id, messages=llama_messages)
        print(f"LLaMA Model Output: {llama_output}")


    asyncio.run(main())
