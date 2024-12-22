# Standard library imports
import os
from typing import List, Dict, Annotated


class SpeakerTimestampReader:
    """
    A class to read and parse speaker timestamps from an RTTM file.

    Attributes
    ----------
    rttm_path : str
        Path to the RTTM file containing speaker timestamps.

    Methods
    -------
    read_speaker_timestamps()
        Reads the RTTM file and extracts speaker timestamps.

    Parameters
    ----------
    rttm_path : str
        Path to the RTTM file containing speaker timestamps.

    Raises
    ------
    FileNotFoundError
        If the RTTM file does not exist at the specified path.

    """

    def __init__(self, rttm_path: str):
        """
        Initializes the SpeakerTimestampReader with the path to an RTTM file.

        Parameters
        ----------
        rttm_path : str
            Path to the RTTM file containing speaker timestamps.

        Raises
        ------
        FileNotFoundError
            If the RTTM file does not exist at the specified path.
        """
        if not os.path.isfile(rttm_path):
            raise FileNotFoundError(f"RTTM file not found at: {rttm_path}")
        self.rttm_path = rttm_path

    def read_speaker_timestamps(self) -> List[List[float]]:
        """
        Reads the RTTM file and extracts speaker timestamps.

        Returns
        -------
        List[List[float]]
            A list where each sublist contains [start_time, end_time, speaker_label].

        Notes
        -----
        - The times are converted to milliseconds.
        - Lines with invalid data are skipped.

        Examples
        --------
        >>> reader = SpeakerTimestampReader("path/to/rttm_file.rttm")
        >>> timestamps = reader.read_speaker_timestamps()
        Speaker_Timestamps: [[0.0, 2000.0, 1], [2100.0, 4000.0, 2]]
        """
        speaker_ts = []
        with open(self.rttm_path) as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.strip().split()
                try:
                    if len(line_list) < 8:
                        print(f"Skipping line due to unexpected format: {line.strip()}")
                        continue

                    start_time = float(line_list[3]) * 1000
                    duration = float(line_list[4]) * 1000
                    end_time = start_time + duration

                    speaker_label_str = line_list[7]
                    speaker_label = int(speaker_label_str.split("_")[-1])

                    speaker_ts.append([start_time, end_time, speaker_label])
                except (ValueError, IndexError) as e:
                    print(f"Skipping line due to parsing error: {line.strip()} - {e}")
                    continue

        print(f"Speaker_Timestamps: {speaker_ts}")
        return speaker_ts


class TranscriptWriter:
    """
    A class to write speaker-aware transcripts in plain text or SRT formats.

    Methods
    -------
    write_transcript(sentences_speaker_mapping, file_path)
        Writes the speaker-aware transcript to a text file.
    write_srt(sentences_speaker_mapping, file_path)
        Writes the speaker-aware transcript to an SRT file format.
    """

    def __init__(self):
        """
        Initializes the TranscriptWriter.
        """
        pass

    @staticmethod
    def write_transcript(sentences_speaker_mapping: List[Dict], file_path: str):
        """
        Writes the speaker-aware transcript to a text file.

        Parameters
        ----------
        sentences_speaker_mapping : List[Dict]
            List of sentences with speaker labels, where each dictionary contains:
            - "speaker": Speaker label (e.g., Speaker 1, Speaker 2).
            - "text": Text of the spoken sentence.
        file_path : str
            Path to the output text file.

        Examples
        --------
        >>> sentences_speaker_map = [{"speaker": "Speaker 1", "text": "Hello."},
                                         {"speaker": "Speaker 2", "text": "Hi there."}]
        >>> TranscriptWriter.write_transcript(sentences_speaker_mapping, "output.txt")
        """
        with open(file_path, "w", encoding="utf-8") as f:
            previous_speaker = sentences_speaker_mapping[0]["speaker"]
            f.write(f"{previous_speaker}: ")

            for sentence_dict in sentences_speaker_mapping:
                speaker = sentence_dict["speaker"]
                sentence = sentence_dict["text"].strip()

                if speaker != previous_speaker:
                    f.write(f"\n\n{speaker}: ")
                    previous_speaker = speaker

                f.write(sentence + " ")

    @staticmethod
    def write_srt(sentences_speaker_mapping: List[Dict], file_path: str):
        """
        Writes the speaker-aware transcript to an SRT file format.

        Parameters
        ----------
        sentences_speaker_mapping : List[Dict]
            List of sentences with speaker labels and timestamps, where each dictionary contains:
            - "start_time": Start time of the sentence in milliseconds.
            - "end_time": End time of the sentence in milliseconds.
            - "speaker": Speaker label.
            - "text": Text of the spoken sentence.
        file_path : str
            Path to the output SRT file.

        Notes
        -----
        The function formats timestamps in the HH:MM:SS,mmm format for SRT.

        Examples
        --------
        >>> sentences_speaker_map = [{"start_time": 0, "end_time": 2000,
                                          "speaker": "Speaker 1", "text": "Hello."}]
        >>> TranscriptWriter.write_srt(sentences_speaker_mapping, "output.srt")
        """

        def format_timestamp(milliseconds: Annotated[float, "Time in milliseconds"]) -> Annotated[
            str, "Formatted timestamp in HH:MM:SS,mmm"]:
            """
            Converts a time value in milliseconds to an SRT timestamp format.

            This function takes a time value in milliseconds and formats it into
            the standard SRT (SubRip Subtitle) timestamp format: `HH:MM:SS,mmm`.

            Parameters
            ----------
            milliseconds : float
                Time value in milliseconds to be converted.

            Returns
            -------
            str
                A string representing the time in `HH:MM:SS,mmm` format.

            Raises
            ------
            ValueError
                If the input time is negative.

            Examples
            --------
            >>> format_timestamp(3723001)
            '01:02:03,001'
            >>> format_timestamp(0)
            '00:00:00,000'
            >>> format_timestamp(59_999.9)
            '00:00:59,999'

            Notes
            -----
            The function ensures the correct zero-padding for hours, minutes,
            seconds, and milliseconds to meet the SRT format requirements.
            """
            if milliseconds < 0:
                raise ValueError("Time in milliseconds cannot be negative.")

            hours = int(milliseconds // 3_600_000)
            minutes = int((milliseconds % 3_600_000) // 60_000)
            seconds = int((milliseconds % 60_000) // 1_000)
            milliseconds = int(milliseconds % 1_000)

            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        with open(file_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(sentences_speaker_mapping, start=1):
                start_time = format_timestamp(segment['start_time'])
                end_time = format_timestamp(segment['end_time'])
                speaker = segment['speaker']
                text = segment['text'].strip().replace('-->', '->')

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{speaker}: {text}\n\n")


if __name__ == "__main__":
    example_rttm_path = "example.rttm"
    try:
        timestamp_reader = SpeakerTimestampReader(example_rttm_path)
        extracted_speaker_timestamps = timestamp_reader.read_speaker_timestamps()
    except FileNotFoundError as file_error:
        print(file_error)

    example_sentences_mapping = [
        {"speaker": "Speaker 1", "text": "Hello there.", "start_time": 0, "end_time": 2000},
        {"speaker": "Speaker 2", "text": "How are you?", "start_time": 2100, "end_time": 4000},
    ]
    transcript_writer = TranscriptWriter()
    transcript_writer.write_transcript(example_sentences_mapping, "output.txt")
    transcript_writer.write_srt(example_sentences_mapping, "output.srt")
