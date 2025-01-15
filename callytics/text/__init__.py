from .model import ModelRegistry, LLaMAModel, OpenAIModel, AzureOpenAIModel

ModelRegistry.register("llama", LLaMAModel)
ModelRegistry.register("openai", OpenAIModel)
ModelRegistry.register("azure_openai", AzureOpenAIModel)