# Standard library imports
import os
import logging
import subprocess
from typing import Annotated

# Related third party imports
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO)


class DialogueDetecting:
    """
    Class for detecting dialogue in audio files using speaker diarization.

    This class processes audio files by dividing them into chunks, applying a
    pre-trained speaker diarization model, and detecting if there are multiple
    speakers in the audio.

    Parameters
    ----------
    pipeline_model : str, optional
        Name of the pre-trained diarization model. Defaults to "pyannote/speaker-diarization".
    chunk_duration : int, optional
        Duration of each chunk in seconds. Defaults to 5.
    sample_rate : int, optional
        Sampling rate for the processed audio chunks. Defaults to 16000.
    channels : int, optional
        Number of audio channels. Defaults to 1.
    delete_original : bool, optional
        If True, deletes the original audio file when no dialogue is detected. Defaults to False.
    skip_if_no_dialogue : bool, optional
        If True, skips further processing if no dialogue is detected. Defaults to False.
    temp_dir : str, optional
        Directory for temporary chunk files. Defaults to ".temp".

    Attributes
    ----------
    pipeline : Pipeline
        Instance of the PyAnnote pipeline for speaker diarization.
    """

    def __init__(self,
                 pipeline_model: str = "pyannote/speaker-diarization",
                 chunk_duration: int = 5,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 delete_original: bool = False,
                 skip_if_no_dialogue: bool = False,
                 temp_dir: str = ".temp"):
        self.pipeline_model = pipeline_model
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.delete_original = delete_original
        self.skip_if_no_dialogue = skip_if_no_dialogue
        self.temp_dir = temp_dir
        self.pipeline = Pipeline.from_pretrained(pipeline_model)

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    @staticmethod
    def get_audio_duration(audio_file: Annotated[str, "Path to the audio file"]) -> Annotated[
        float, "Duration of the audio in seconds"]:
        """
        Get the duration of an audio file in seconds.

        Parameters
        ----------
        audio_file : str
            Path to the audio file.

        Returns
        -------
        float
            Duration of the audio file in seconds.

        Examples
        --------
        >>> DialogueDetecting.get_audio_duration("example.wav")
        120.5
        """
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_file],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())

    def create_chunk(self, audio_file: str, chunk_file: str, start_time: float, end_time: float):
        """
        Create a chunk of the audio file.

        Parameters
        ----------
        audio_file : str
            Path to the original audio file.
        chunk_file : str
            Path to save the generated chunk file.
        start_time : float
            Start time of the chunk in seconds.
        end_time : float
            End time of the chunk in seconds.
        """
        duration = end_time - start_time
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", audio_file,
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-f", "wav",
            chunk_file
        ], check=True)

    def process_chunk(self, chunk_file: Annotated[str, "Path to the chunk file"]) -> Annotated[
        set, "Set of detected speaker labels"]:
        """
        Process a single chunk of audio to detect speakers.

        Parameters
        ----------
        chunk_file : str
            Path to the chunk file.

        Returns
        -------
        set
            Set of detected speaker labels in the chunk.
        """
        diarization = self.pipeline(chunk_file)
        speakers_in_chunk = set()
        for segment, track, label in diarization.itertracks(yield_label=True):
            speakers_in_chunk.add(label)
        return speakers_in_chunk

    def process(self, audio_file: Annotated[str, "Path to the input audio file"]) -> Annotated[
        bool, "True if dialogue detected, False otherwise"]:
        """
        Process the audio file to detect dialogue.

        Parameters
        ----------
        audio_file : str
            Path to the audio file.

        Returns
        -------
        bool
            True if at least two speakers are detected, False otherwise.

        Examples
        --------
        >>> dialogue_detector = DialogueDetecting()
        >>> dialogue_detector.process("example.wav")
        True
        """
        total_duration = self.get_audio_duration(audio_file)
        num_chunks = int(total_duration // self.chunk_duration) + 1

        speakers_detected = set()
        chunk_files = []

        try:
            for i in range(num_chunks):
                start_time = i * self.chunk_duration
                end_time = min(float((i + 1) * self.chunk_duration), total_duration)

                if end_time - start_time < 1.0:
                    logging.info("Last chunk is too short to process.")
                    break

                chunk_file = os.path.join(self.temp_dir, f"chunk_{i}.wav")
                chunk_files.append(chunk_file)
                logging.info(f"Creating chunk: {chunk_file}")
                self.create_chunk(audio_file, chunk_file, start_time, end_time)

                logging.info(f"Processing chunk: {chunk_file}")
                chunk_speakers = self.process_chunk(chunk_file)
                speakers_detected.update(chunk_speakers)

                if len(speakers_detected) >= 2:
                    logging.info("At least two speakers detected, stopping.")
                    return True

            if len(speakers_detected) < 2:
                logging.info("No dialogue detected or only one speaker found.")
                if self.delete_original:
                    logging.info(f"No dialogue found. Deleting original file: {audio_file}")
                    os.remove(audio_file)
                if self.skip_if_no_dialogue:
                    logging.info("Skipping further processing due to lack of dialogue.")
                    return False

        finally:
            logging.info("Cleaning up temporary chunk files.")
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)

            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)

        return len(speakers_detected) >= 2


if __name__ == "__main__":
    processor = DialogueDetecting(delete_original=True)
    audio_path = ".data/example/kafkasya.mp3"
    process_result = processor.process(audio_path)
    print("Dialogue detected:", process_result)
