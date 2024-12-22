# Standard library imports
import os
from typing import Annotated, Dict, Any

# Related third-party imports
import yaml


class PromptManager:
    """
    A class to manage prompts loaded from a YAML configuration file.

    This class provides methods to load prompts from a YAML file and retrieve
    them with optional formatting.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML configuration file. If not provided, the default
        path is set to `../../config/prompts.yaml` relative to the script's
        directory.

    Attributes
    ----------
    config_path : str
        Path to the YAML configuration file.
    prompts : dict
        Dictionary of prompts loaded from the YAML file.
    """

    def __init__(self, config_path: Annotated[str, "Path to the YAML configuration file"] = None):
        """
        Initializes the PromptManager with a specified or default configuration path.

        Parameters
        ----------
        config_path : str, optional
            Path to the YAML configuration file. Defaults to None.
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../../config/prompts.yaml"
        )
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Annotated[Dict[str, Any], "Loaded prompts from YAML file"]:
        """
        Load prompts from the YAML file.

        This method reads the YAML file specified by `self.config_path` and parses its contents.

        Returns
        -------
        dict
            Dictionary containing the prompts.

        Raises
        ------
        FileNotFoundError
            If the specified YAML file does not exist.

        Examples
        --------
        >>> manager = PromptManager("config/prompts.yaml")
        >>> prompts = manager._load_prompts()
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"YAML file not found: {self.config_path}")

        with open(self.config_path) as file:
            loaded_prompts = yaml.safe_load(file)
            if not isinstance(loaded_prompts, dict):
                raise TypeError(f"Expected dictionary from YAML, got {type(loaded_prompts).__name__}.")
            return loaded_prompts

    def get_prompt(
            self,
            prompt_name: Annotated[str, "Name of the prompt to retrieve"],
            **kwargs: Annotated[dict, "Keyword arguments for formatting the prompt"]
    ) -> Annotated[Any, "Formatted prompt (str or dict)"]:
        """
        Retrieve and format a prompt by its name.

        This method fetches the prompt template identified by `prompt_name` from the loaded prompts
        and formats any placeholders within the prompt using the provided keyword arguments.

        Parameters
        ----------
        prompt_name : str
            Name of the prompt to retrieve.
        **kwargs : Any
            Keyword arguments to format the prompt strings.

        Raises
        ------
        ValueError
            If the specified prompt name does not exist in the loaded prompts.

        Returns
        -------
        dict
            Dictionary containing the formatted prompt with all placeholders replaced by provided values.
        """
        if not isinstance(self.prompts, dict):
            raise TypeError(f"Internal error: self.prompts is not a dictionary but {type(self.prompts).__name__}.")

        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found.")

        prompt = self.prompts[prompt_name]

        if isinstance(prompt, dict):
            formatted_prompt = {}
            for key, value in prompt.items():
                if isinstance(value, str):
                    formatted_prompt[key] = value.format(**kwargs)
                else:
                    formatted_prompt[key] = value
            return formatted_prompt

        if isinstance(prompt, str):
            return prompt.format(**kwargs)

        raise TypeError(f"Unexpected prompt type: {type(prompt).__name__}")


if __name__ == "__main__":
    try:
        prompt_manager = PromptManager()

        formatted_prompt_ = prompt_manager.get_prompt(
            "greeting",
            name="Ahmet",
            day="Pazartesi"
        )

        print("Formatted Prompt:", formatted_prompt_)

    except Exception as e:
        print(f"Error: {e}")