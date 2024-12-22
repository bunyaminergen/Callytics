# Standard library imports
import os
from typing import Annotated, List, Dict

# Related third-party imports
import torch
from faster_whisper import decode_audio
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)


class ForcedAligner:
    """
    ForcedAligner is a class for aligning audio to a provided transcript using a pre-trained alignment model.

    Attributes
    ----------
    device : str
        Device to run the model on ('cuda' for GPU or 'cpu').
    alignment_model : torch.nn.Module
        The pre-trained alignment model.
    alignment_tokenizer : Any
        Tokenizer for processing text in alignment.

    Methods
    -------
    align(audio_path, transcript, language, batch_size)
        Aligns audio with a transcript and returns word-level timing information.
    """

    def __init__(self, device: Annotated[str, "Device for model ('cuda' or 'cpu')"] = None):
        """
        Initialize the ForcedAligner with the specified device.

        Parameters
        ----------
        device : str, optional
            Device for running the model, by default 'cuda' if available, otherwise 'cpu'.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=torch.float16 if self.device == 'cuda' else torch.float32,
        )

    def align(
            self,
            audio_path: Annotated[str, "Path to the audio file"],
            transcript: Annotated[str, "Transcript of the audio content"],
            language: Annotated[str, "Language of the transcript"] = 'en',
            batch_size: Annotated[int, "Batch size for emission generation"] = 8,
    ) -> Annotated[List[Dict[str, float]], "List of word alignment data with timestamps"]:
        """
        Aligns audio with a transcript and returns word-level timing information.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        transcript : str
            Transcript text corresponding to the audio.
        language : str, optional
            Language code for the transcript, default is 'en' (English).
        batch_size : int, optional
            Batch size for generating emissions, by default 8.

        Returns
        -------
        List[Dict[str, float]]
            A list of dictionaries containing word timing information.

        Raises
        ------
        FileNotFoundError
            If the specified audio file does not exist.

        Examples
        --------
        >>> aligner = ForcedAligner()
        >>> aligner.align("path/to/audio.wav", "hello world")
        [{'word': 'hello', 'start': 0.0, 'end': 0.5}, {'word': 'world', 'start': 0.6, 'end': 1.0}]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"The audio file at path '{audio_path}' was not found."
            )

        speech_array = torch.from_numpy(decode_audio(audio_path))

        emissions, stride = generate_emissions(
            self.alignment_model,
            speech_array.to(self.alignment_model.dtype).to(self.alignment_model.device),
            batch_size=batch_size,
        )

        tokens_starred, text_starred = preprocess_text(
            transcript,
            romanize=True,
            language=language,
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self.alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)

        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        if self.device == 'cuda':
            del self.alignment_model
            torch.cuda.empty_cache()

        print(f"Word_Timestamps: {word_timestamps}")

        return word_timestamps


if __name__ == "__main__":

    forced_aligner = ForcedAligner()
    try:
        path = "example_audio.wav"
        audio_transcript = "This is a test transcript."
        word_timestamp = forced_aligner.align(path, audio_transcript)
        print(word_timestamp)
    except FileNotFoundError as e:
        print(e)