# Standard library imports
import os
import warnings
from typing import Annotated, Optional

# Related third-party imports
import demucs.separate


class DemucsVocalSeparator:
    """
    A class for separating vocals from an audio file using the Demucs model.

    This class utilizes the Demucs model to separate specified audio stems (e.g., vocals) from an input audio file.
    It supports saving the separated outputs to a specified directory.

    Attributes
    ----------
    model_name : str
        Name of the Demucs model to use for separation.
    two_stems : str
        The stem to isolate (e.g., "vocals").

    Methods
    -------
    separate_vocals(audio_file: str, output_dir: str) -> Optional[str]
        Separates vocals (or other specified stem) from the audio file and returns the path to the separated file.

    """

    def __init__(
            self,
            model_name: Annotated[str, "Demucs model name to use for separation"] = "htdemucs",
            two_stems: Annotated[str, "Stem to isolate (e.g., vocals, drums)"] = "vocals"
    ):
        """
        Initializes the DemucsVocalSeparator with the given parameters.

        Parameters
        ----------
        model_name : str, optional
            Name of the Demucs model to use for separation (default is "htdemucs").
        two_stems : str, optional
            The stem to isolate (default is "vocals").
        """
        self.model_name = model_name
        self.two_stems = two_stems

    def separate_vocals(self, audio_file: str, output_dir: str) -> Optional[str]:
        """
        Separates vocals (or other specified stem) from the audio file.

        This method invokes the Demucs model to isolate a specified audio stem (e.g., vocals).
        The output is saved in WAV format in the specified output directory.

        Parameters
        ----------
        audio_file : str
            Path to the input audio file.
        output_dir : str
            Directory where the separated files will be saved.

        Returns
        -------
        Optional[str]
            Path to the separated vocal file if successful, or the original audio file path if not.

        Raises
        ------
        Warning
            If vocal separation fails or the separated file is not found.

        Examples
        --------
        >>> separator = DemucsVocalSeparator()
        >>> vocal_path = separator.separate_vocals("path/to/audio/file.mp3", "output_dir")
        Vocal separation successful! Outputs saved in WAV format at 'output_dir' directory.
        """
        demucs_args = [
            "--two-stems", self.two_stems,
            "-n", self.model_name,
            "-o", output_dir,
            audio_file
        ]

        try:
            demucs.separate.main(demucs_args)
            print(f"Vocal separation successful! Outputs saved in WAV format at '{output_dir}' directory.")

            output_path = os.path.join(
                output_dir, self.model_name,
                os.path.splitext(os.path.basename(audio_file))[0]
            )
            vocal_file = os.path.join(output_path, f"{self.two_stems}.wav")

            if os.path.exists(vocal_file):
                return vocal_file
            else:
                print("Separated vocal file not found. Returning original audio file path.")
                warnings.warn("Vocal separation was unsuccessful; using the original audio file.", stacklevel=2)
                return audio_file

        except Exception as e:
            print(f"An error occurred during vocal separation: {e}")
            warnings.warn("Vocal separation failed; proceeding with the original audio file.", stacklevel=2)
            return audio_file


if __name__ == "__main__":
    file = "example_audio.mp3"
    output_directory = "separated_outputs"
    vocal_separator = DemucsVocalSeparator()
    separated_file_path = vocal_separator.separate_vocals(file, output_directory)
    print(f"Separated file path: {separated_file_path}")
