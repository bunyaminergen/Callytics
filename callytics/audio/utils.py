# Standard library imports
import warnings
from typing import List, Dict, Union


class TokenizerUtils:
    """
    Utility class for handling token-related operations, particularly for identifying tokens
    that contain numerals or specific symbols.

    This class includes an __init__ method for completeness, but it does not perform any
    initialization since the class is intended to be used as a static utility class.

    Methods
    -------
    find_numeral_symbol_tokens(tokenizer)
        Returns a list of token IDs that include numerals or symbols like '%', '$', or '£'.
    """

    def __init__(self):
        """Initialize the TokenizerUtils class. This method is present for completeness."""
        pass

    @staticmethod
    def find_numeral_symbol_tokens(tokenizer) -> List[int]:
        """
        Identifies tokens that contain numerals or certain symbols in the tokenizer vocabulary.

        Parameters
        ----------
        tokenizer : Any
            Tokenizer object with a 'get_vocab' method, typically from Hugging Face's tokenizer library.

        Returns
        -------
        List[int]
            List of token IDs for tokens that contain numerals or symbols.

        Examples
        --------
        >>> TokenizerUtils.find_numeral_symbol_tokens(tokenizer)
        [-1, 123, 456, 789]
        """
        numeral_symbol_tokens = [-1]
        for token, token_id in tokenizer.get_vocab().items():
            if any(c in "0123456789%$£" for c in token):
                numeral_symbol_tokens.append(token_id)
        return numeral_symbol_tokens


class Formatter:
    """
    A utility class for formatting audio-related data, such as sentence-speaker mappings.

    Methods
    -------
    add_indices_to_ssm(ssm: List[Dict], reference_length: int = None) -> List[Dict]:
        Adds an index key to each item in the SSM list and checks for length mismatches with a reference.
    format_ssm_as_dialogue(
        ssm: List[Dict],
        print_output: bool = False,
        return_dict: bool = False
    ) -> Union[str, Dict[str, List[str]]]:
        Formats sentence-speaker mappings into a readable dialogue format and optionally prints it or returns a
        dictionary grouped by speakers.
    """

    @staticmethod
    def add_indices_to_ssm(ssm: List[Dict], reference_length: int = None) -> List[Dict]:
        """
        Adds an index key to each item in the SSM list and optionally checks for length mismatches with a reference
        length.

        Parameters
        ----------
        ssm : List[Dict]
            The final SSM data.
        reference_length : int, optional
            A reference length to compare the SSM length against, default is None.

        Returns
        -------
        List[Dict]
            The SSM data with added index keys and any necessary adjustments.
        """
        if reference_length is not None and len(ssm) != reference_length:
            warnings.warn(
                f"Mismatch: SSM Length = {len(ssm)}, Reference Length = {reference_length}. "
                f"Adjusting to match lengths...",
                UserWarning,
            )

        for idx, item in enumerate(ssm):
            item["index"] = idx

        if reference_length is not None:
            if len(ssm) > reference_length:
                ssm = ssm[:reference_length]
            elif len(ssm) < reference_length:
                for i in range(len(ssm), reference_length):
                    ssm.append({
                        "index": i,
                        "speaker": "Unknown",
                        "start_time": None,
                        "end_time": None,
                        "text": "[Placeholder]"
                    })

        return ssm

    @staticmethod
    def format_ssm_as_dialogue(
            ssm: List[Dict],
            print_output: bool = False,
            return_dict: bool = False
    ) -> Union[str, Dict[str, List[str]]]:
        """
        Formats the sentence-speaker mapping (ssm) as a dialogue and optionally prints the result or returns it as a
        dictionary grouped by speakers.

        Parameters
        ----------
        ssm : List[Dict]
            List of sentences with speaker labels.
        print_output : bool, optional
            Whether to print the formatted dialogue, default is False.
        return_dict : bool, optional
            Whether to return the response as a dictionary grouped by speakers, default is False.

        Returns
        -------
        Union[str, Dict[str, List[str]]]
            If `return_dict` is True, returns a dictionary with speakers as keys and lists of their sentences as values.
            Otherwise, returns the formatted dialogue string.
        """
        dialogue_dict: Dict[str, List[str]] = {}

        for sentence in ssm:
            speaker = sentence['speaker']
            text = sentence['text'].strip()

            if speaker in dialogue_dict:
                dialogue_dict[speaker].append(text)
            else:
                dialogue_dict[speaker] = [text]

        if print_output:
            print("Formatted Dialogue:")
            for speaker, texts in dialogue_dict.items():
                for text in texts:
                    print(f"{speaker}: {text}")
                print()

        if return_dict:
            return dialogue_dict

        formatted_dialogue = "\n\n".join(
            [f"{speaker}: {text}" for speaker, texts in dialogue_dict.items() for text in texts]
        )
        return formatted_dialogue


if __name__ == "__main__":
    # noinspection PyMissingOrEmptyDocstring
    class DummyTokenizer:
        @staticmethod
        def get_vocab():
            return {
                "hello": 1,
                "world": 2,
                "100%": 3,
                "$value": 4,
                "item_123": 5,
                "£price": 6
            }


    dummy_tokenizer = DummyTokenizer()
    numeral_tokens = TokenizerUtils.find_numeral_symbol_tokens(dummy_tokenizer)
    print(f"Numeral and symbol tokens: {numeral_tokens}")

    speaker_sentence_mapping = [
        {"speaker": "Speaker 1", "text": "Hello, how are you?"},
        {"speaker": "Speaker 2", "text": "I'm fine, thank you! And you?"},
        {"speaker": "Speaker 1", "text": "I'm doing great, thanks for asking."}
    ]

    formatted_dialogue_str = Formatter.format_ssm_as_dialogue(speaker_sentence_mapping, print_output=True)
    print(f"Formatted Dialogue:\n{formatted_dialogue_str}")
