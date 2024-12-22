# Standard library imports
import os
import re
import json
from io import TextIOWrapper
from typing import Annotated, Optional, Tuple, List, Dict

# Related third party imports
import torch
import faster_whisper
from pydub import AudioSegment
from deepmultilingualpunctuation import PunctuationModel

# Local imports
from src.audio.utils import TokenizerUtils


class AudioProcessor:
    """
    A class to handle various audio processing tasks, such as conversion,
    trimming, merging, and audio transformations.

    Parameters
    ----------
    audio_path : str
        Path to the audio file to process.
    temp_dir : str, optional
        Directory for storing temporary files. Defaults to ".temp".

    Attributes
    ----------
    audio_path : str
        Path to the input audio file.
    temp_dir : str
        Path to the temporary directory for processed files.
    mono_audio_path : Optional[str]
        Path to the mono audio file after conversion.

    Methods
    -------
    convert_to_mono()
        Converts the audio file to mono.
    get_duration()
        Gets the duration of the audio file in seconds.
    change_format(new_format)
        Converts the audio file to a new format.
    trim_audio(start_time, end_time)
        Trims the audio file to the specified time range.
    adjust_volume(change_in_db)
        Adjusts the volume of the audio file.
    get_channels()
        Gets the number of audio channels.
    fade_in_out(fade_in_duration, fade_out_duration)
        Applies fade-in and fade-out effects to the audio.
    merge_audio(other_audio_path)
        Merges the current audio with another audio file.
    split_audio(chunk_duration)
        Splits the audio file into chunks of a specified duration.
    create_manifest(manifest_path)
        Creates a manifest file containing metadata about the audio.
    """

    def __init__(
            self,
            audio_path: Annotated[str, "Path to the audio file"],
            temp_dir: Annotated[str, "Directory for temporary processed files"] = ".temp"
    ) -> None:
        if not isinstance(audio_path, str):
            raise TypeError("Expected 'audio_path' to be a string.")
        if not isinstance(temp_dir, str):
            raise TypeError("Expected 'temp_dir' to be a string.")

        self.audio_path = audio_path
        self.temp_dir = temp_dir
        self.mono_audio_path = None
        os.makedirs(temp_dir, exist_ok=True)

    def convert_to_mono(self) -> Annotated[str, "Path to the mono audio file"]:
        """
        Convert the audio file to mono.

        Returns
        -------
        str
            Path to the mono audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> mono_path = processor.convert_to_mono()
        >>> isinstance(mono_path, str)
        True
        """
        sound = AudioSegment.from_file(self.audio_path)
        mono_sound = sound.set_channels(1)
        self.mono_audio_path = os.path.join(self.temp_dir, "mono_file.wav")
        mono_sound.export(self.mono_audio_path, format="wav")
        return self.mono_audio_path

    def get_duration(self) -> Annotated[float, "Audio duration in seconds"]:
        """
        Get the duration of the audio file.

        Returns
        -------
        float
            Duration of the audio in seconds.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> duration = processor.get_duration()
        >>> isinstance(duration, float)
        True
        """
        sound = AudioSegment.from_file(self.audio_path)
        return len(sound) / 1000.0

    def change_format(
            self, new_format: Annotated[str, "New audio format"]
    ) -> Annotated[str, "Path to converted audio file"]:
        """
        Convert the audio file to a new format.

        Parameters
        ----------
        new_format : str
            Desired format for the output audio file.

        Returns
        -------
        str
            Path to the converted audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> converted_path = processor.change_format("mp3")
        >>> isinstance(converted_path, str)
        True
        """
        if not isinstance(new_format, str):
            raise TypeError("Expected 'new_format' to be a string.")

        sound = AudioSegment.from_file(self.audio_path)
        output_path = os.path.join(self.temp_dir, f"converted_file.{new_format}")
        sound.export(output_path, format=new_format)
        return output_path

    def trim_audio(
            self, start_time: Annotated[float, "Start time in seconds"],
            end_time: Annotated[float, "End time in seconds"]
    ) -> Annotated[str, "Path to trimmed audio file"]:
        """
        Trim the audio file to the specified duration.

        Parameters
        ----------
        start_time : float
            Start time in seconds.
        end_time : float
            End time in seconds.

        Returns
        -------
        str
            Path to the trimmed audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> trimmed_path = processor.trim_audio(0.0, 10.0)
        >>> isinstance(trimmed_path, str)
        True
        """
        if not isinstance(start_time, (int, float)):
            raise TypeError("Expected 'start_time' to be a float or int.")
        if not isinstance(end_time, (int, float)):
            raise TypeError("Expected 'end_time' to be a float or int.")

        sound = AudioSegment.from_file(self.audio_path)
        trimmed_audio = sound[start_time * 1000:end_time * 1000]
        trimmed_audio_path = os.path.join(self.temp_dir, "trimmed_file.wav")
        trimmed_audio.export(trimmed_audio_path, format="wav")
        return trimmed_audio_path

    def adjust_volume(
            self, change_in_db: Annotated[float, "Volume change in dB"]
    ) -> Annotated[str, "Path to volume-adjusted audio file"]:
        """
        Adjust the volume of the audio file.

        Parameters
        ----------
        change_in_db : float
            Volume change in decibels.

        Returns
        -------
        str
            Path to the volume-adjusted audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> adjusted_path = processor.adjust_volume(5.0)
        >>> isinstance(adjusted_path, str)
        True
        """
        if not isinstance(change_in_db, (int, float)):
            raise TypeError("Expected 'change_in_db' to be a float or int.")

        sound = AudioSegment.from_file(self.audio_path)
        adjusted_audio = sound + change_in_db
        adjusted_audio_path = os.path.join(self.temp_dir, "adjusted_volume.wav")
        adjusted_audio.export(adjusted_audio_path, format="wav")
        return adjusted_audio_path

    def get_channels(self) -> Annotated[int, "Number of channels"]:
        """
        Get the number of audio channels.

        Returns
        -------
        int
            Number of audio channels.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> channels = processor.get_channels()
        >>> isinstance(channels, int)
        True
        """
        sound = AudioSegment.from_file(self.audio_path)
        return sound.channels

    def fade_in_out(
            self, fade_in_duration: Annotated[float, "Fade-in duration in seconds"],
            fade_out_duration: Annotated[float, "Fade-out duration in seconds"]
    ) -> Annotated[str, "Path to faded audio file"]:
        """
        Apply fade-in and fade-out effects to the audio file.

        Parameters
        ----------
        fade_in_duration : float
            Duration of the fade-in effect in seconds.
        fade_out_duration : float
            Duration of the fade-out effect in seconds.

        Returns
        -------
        str
            Path to the faded audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> faded_path = processor.fade_in_out(1.0, 2.0)
        >>> isinstance(faded_path, str)
        True
        """
        if not isinstance(fade_in_duration, (int, float)):
            raise TypeError("Expected 'fade_in_duration' to be a float or int.")
        if not isinstance(fade_out_duration, (int, float)):
            raise TypeError("Expected 'fade_out_duration' to be a float or int.")

        sound = AudioSegment.from_file(self.audio_path)
        faded_audio = sound.fade_in(fade_in_duration * 1000).fade_out(fade_out_duration * 1000)
        faded_audio_path = os.path.join(self.temp_dir, "faded_audio.wav")
        faded_audio.export(faded_audio_path, format="wav")
        return faded_audio_path

    def merge_audio(
            self, other_audio_path: Annotated[str, "Path to other audio file"]
    ) -> Annotated[str, "Path to merged audio file"]:
        """
        Merge the current audio file with another audio file.

        Parameters
        ----------
        other_audio_path : str
            Path to the other audio file.

        Returns
        -------
        str
            Path to the merged audio file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> merged_path = processor.merge_audio("other_example.wav")
        >>> isinstance(merged_path, str)
        True
        """
        if not isinstance(other_audio_path, str):
            raise TypeError("Expected 'other_audio_path' to be a string.")

        sound1 = AudioSegment.from_file(self.audio_path)
        sound2 = AudioSegment.from_file(other_audio_path)
        merged_audio = sound1 + sound2
        merged_audio_path = os.path.join(self.temp_dir, "merged_audio.wav")
        merged_audio.export(merged_audio_path, format="wav")
        return merged_audio_path

    def split_audio(
            self, chunk_duration: Annotated[float, "Chunk duration in seconds"]
    ) -> Annotated[List[str], "Paths to audio chunks"]:
        """
        Split the audio file into chunks of the specified duration.

        Parameters
        ----------
        chunk_duration : float
            Duration of each chunk in seconds.

        Returns
        -------
        List[str]
            Paths to the generated audio chunks.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> chunks = processor.split_audio(10.0)
        >>> isinstance(chunks, list)
        True
        """
        if not isinstance(chunk_duration, (int, float)):
            raise TypeError("Expected 'chunk_duration' to be a float or int.")

        sound = AudioSegment.from_file(self.audio_path)
        chunk_paths = []

        for i in range(0, len(sound), int(chunk_duration * 1000)):
            chunk = sound[i:i + int(chunk_duration * 1000)]
            chunk_path = os.path.join(self.temp_dir, f"chunk_{i // 1000}.wav")
            chunk.export(chunk_path, format="wav")
            chunk_paths.append(chunk_path)

        return chunk_paths

    def create_manifest(
            self,
            manifest_path: Annotated[str, "Manifest file path"]
    ) -> None:
        """
        Create a manifest file containing metadata about the audio file.

        Parameters
        ----------
        manifest_path : str
            Path to the manifest file.

        Examples
        --------
        >>> processor = AudioProcessor("example.wav")
        >>> processor.create_manifest("manifest.json")
        """
        duration = self.get_duration()
        manifest_entry = {
            "audio_filepath": self.audio_path,
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None
        }
        with open(manifest_path, 'w', encoding='utf-8') as f:  # type: TextIOWrapper
            json.dump(manifest_entry, f)


class Transcriber:
    """
    A class for transcribing audio files using a pre-trained Whisper model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to load. Defaults to 'large-v3'.
    device : str, optional
        Device to use for model inference ('cpu' or 'cuda'). Defaults to 'cpu'.
    compute_type : str, optional
        Data type for model computation ('int8', 'float16', etc.). Defaults to 'int8'.

    Attributes
    ----------
    model : faster_whisper.WhisperModel
        Loaded Whisper model for transcription.
    device : str
        Device used for inference.

    Methods
    -------
    transcribe(audio_path, language=None, suppress_numerals=False)
        Transcribes the audio file into text.
    """

    def __init__(
            self,
            model_name: Annotated[str, "Name of the model to load"] = 'large-v3',
            device: Annotated[str, "Device to use for model inference"] = 'cpu',
            compute_type: Annotated[str, "Data type for model computation, e.g., 'int8' or 'float16'"] = 'int8'
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("Expected 'model_name' to be of type str")
        if not isinstance(device, str):
            raise TypeError("Expected 'device' to be of type str")
        if not isinstance(compute_type, str):
            raise TypeError("Expected 'compute_type' to be of type str")

        self.device = device
        self.model = faster_whisper.WhisperModel(
            model_name, device=device, compute_type=compute_type
        )

    def transcribe(
            self,
            audio_path: Annotated[str, "Path to the audio file to transcribe"],
            language: Annotated[Optional[str], "Language code for transcription, e.g., 'en' for English"] = None,
            suppress_numerals: Annotated[bool, "Whether to suppress numerals in the transcription"] = False
    ) -> Annotated[Tuple[str, dict], "Transcription text and additional information"]:
        """
        Transcribe an audio file into text.

        Parameters
        ----------
        audio_path : str
            Path to the audio file.
        language : str, optional
            Language code for transcription (e.g., 'en' for English).
        suppress_numerals : bool, optional
            Whether to suppress numerals in the transcription. Defaults to False.

        Returns
        -------
        Tuple[str, dict]
            The transcribed text and additional transcription metadata.

        Examples
        --------
        >>> transcriber = Transcriber()
        >>> text, information = transcriber.transcribe("example.wav")
        >>> isinstance(text, str)
        True
        >>> isinstance(info, dict)
        True
        """
        if not isinstance(audio_path, str):
            raise TypeError("Expected 'audio_path' to be of type str")
        if language is not None and not isinstance(language, str):
            raise TypeError("Expected 'language' to be of type str if provided")
        if not isinstance(suppress_numerals, bool):
            raise TypeError("Expected 'suppress_numerals' to be of type bool")

        audio_waveform = faster_whisper.decode_audio(audio_path)
        suppress_tokens = [-1]
        if suppress_numerals:
            suppress_tokens = TokenizerUtils.find_numeral_symbol_tokens(
                self.model.hf_tokenizer
            )

        transcript_segments, info = self.model.transcribe(
            audio_waveform,
            language=language,
            suppress_tokens=suppress_tokens,
            without_timestamps=True,
            vad_filter=True,
            log_progress=True,
        )

        transcript = ''.join(segment.text for segment in transcript_segments)
        info = vars(info)

        if self.device == 'cuda':
            del self.model
            torch.cuda.empty_cache()

        print(transcript, info)

        return transcript, info


class PunctuationRestorer:
    """
    A class for restoring punctuation in transcribed text.

    Parameters
    ----------
    language : str, optional
        Language for punctuation restoration. Defaults to 'en'.

    Attributes
    ----------
    language : str
        Language used for punctuation restoration.
    punct_model : PunctuationModel
        Model for predicting punctuation.
    supported_languages : List[str]
        List of languages supported by the model.

    Methods
    -------
    restore_punctuation(word_speaker_mapping)
        Restores punctuation in the provided text based on word mappings.
    """

    def __init__(self, language: Annotated[str, "Language for punctuation restoration"] = 'en') -> None:
        self.language = language
        self.punct_model = PunctuationModel(model="kredor/punctuate-all")
        self.supported_languages = [
            "en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl",
        ]

    def restore_punctuation(
            self, word_speaker_mapping: Annotated[List[Dict], "List of word-speaker mappings"]
    ) -> Annotated[List[Dict], "Word mappings with restored punctuation"]:
        """
        Restore punctuation for transcribed text.

        Parameters
        ----------
        word_speaker_mapping : List[Dict]
            List of dictionaries containing word and speaker mappings.

        Returns
        -------
        List[Dict]
            Updated list with punctuation restored.

        Examples
        --------
        >>> restorer = PunctuationRestorer()
        >>> mapping = [{"text": "hello"}, {"text": "world"}]
        >>> result = restorer.restore_punctuation(mapping)
        >>> isinstance(result, list)
        True
        >>> "text" in result[0]
        True
        """
        if self.language not in self.supported_languages:
            print(f"Punctuation restoration is not available for {self.language} language.")
            return word_speaker_mapping

        words_list = [word_dict["text"] for word_dict in word_speaker_mapping]
        labeled_words = self.punct_model.predict(words_list)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(word_speaker_mapping, labeled_words):
            word = word_dict["text"]
            if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                word = word.rstrip(".") if word.endswith("..") else word
                word_dict["text"] = word

        return word_speaker_mapping


if __name__ == "__main__":
    sample_audio_path = "sample_audio.wav"
    audio_processor_instance = AudioProcessor(sample_audio_path)

    mono_audio_path = audio_processor_instance.convert_to_mono()
    print(f"Mono audio file saved at: {mono_audio_path}")

    audio_duration = audio_processor_instance.get_duration()
    print(f"Audio duration: {audio_duration} seconds")

    converted_audio_path = audio_processor_instance.change_format("mp3")
    print(f"Converted audio file saved at: {converted_audio_path}")

    audio_path_trimmed = audio_processor_instance.trim_audio(0.0, 10.0)
    print(f"Trimmed audio file saved at: {audio_path_trimmed}")

    volume_adjusted_audio_path = audio_processor_instance.adjust_volume(5.0)
    print(f"Volume adjusted audio file saved at: {volume_adjusted_audio_path}")

    additional_audio_path = "additional_audio.wav"
    merged_audio_output_path = audio_processor_instance.merge_audio(additional_audio_path)
    print(f"Merged audio file saved at: {merged_audio_output_path}")

    audio_chunk_paths = audio_processor_instance.split_audio(10.0)
    print(f"Audio chunks saved at: {audio_chunk_paths}")

    output_manifest_path = "output_manifest.json"
    audio_processor_instance.create_manifest(output_manifest_path)
    print(f"Manifest file saved at: {output_manifest_path}")

    transcriber_instance = Transcriber()
    transcribed_text_output, transcription_metadata = transcriber_instance.transcribe(sample_audio_path)
    print(f"Transcribed Text: {transcribed_text_output}")
    print(f"Transcription Info: {transcription_metadata}")

    word_mapping_example = [
        {"text": "hello"},
        {"text": "world"},
        {"text": "this"},
        {"text": "is"},
        {"text": "a"},
        {"text": "test"}
    ]
    punctuation_restorer_instance = PunctuationRestorer()
    punctuation_restored_mapping = punctuation_restorer_instance.restore_punctuation(word_mapping_example)
    print(f"Restored Mapping: {punctuation_restored_mapping}")
