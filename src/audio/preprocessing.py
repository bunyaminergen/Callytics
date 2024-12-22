# Standard library imports
import os
from typing import Annotated

# Related third-party imports
import librosa
import soundfile as sf
from librosa.feature import rms
from omegaconf import OmegaConf
from noisereduce import reduce_noise

# Local imports
from src.utils.utils import Logger


class Denoiser:
    """
    A class to handle audio denoising using librosa and noisereduce.

    This class provides methods to load noisy audio, apply denoising, and
    save the cleaned output to disk.

    Parameters
    ----------
    config_path : str
        Path to the configuration file that specifies runtime settings.
    output_dir : str, optional
        Directory to save cleaned audio files. Defaults to ".temp".

    Attributes
    ----------
    config : omegaconf.DictConfig
        Loaded configuration data.
    output_dir : str
        Directory to save cleaned audio files.
    """

    def __init__(self, config_path: Annotated[str, "Path to the config file"],
                 output_dir: Annotated[str, "Default directory to save cleaned audio files"] = ".temp") -> None:
        """
        Initialize the Denoiser class.

        Parameters
        ----------
        config_path : str
            Path to the configuration file that specifies runtime settings.
        output_dir : str, optional
            Default directory to save cleaned audio files. Defaults to ".temp".
        """
        self.config = OmegaConf.load(config_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = Logger(name="DenoiserLogger")

    def denoise_audio(
            self,
            input_path: Annotated[str, "Path to the noisy audio file"],
            output_dir: Annotated[str, "Directory to save the cleaned audio file"],
            noise_threshold: Annotated[float, "Noise threshold value to decide if denoising is needed"],
            print_output: Annotated[bool, "Whether to log the process to console"] = False,
    ) -> str:
        """
        Denoise an audio file using noisereduce and librosa.

        Parameters
        ----------
        input_path : str
            Path to the noisy input audio file.
        output_dir : str
            Directory to save the cleaned audio file.
        noise_threshold : float
            Noise threshold value to decide if denoising is needed.
        print_output : bool, optional
            Whether to log the process to the console. Defaults to False.

        Returns
        -------
        str
            Path to the saved audio file if denoising is performed, otherwise the original audio file path.
        """
        self.logger.log(f"Loading: {input_path}", print_output=print_output)

        noisy_waveform, sr = librosa.load(input_path, sr=None)

        noise_level = rms(y=noisy_waveform).mean()
        self.logger.log(f"Calculated noise level: {noise_level}", print_output=print_output)

        if noise_level < noise_threshold:
            self.logger.log("Noise level is below the threshold. Skipping denoising.", print_output=print_output)
            return input_path

        self.logger.log("Denoising process started...", print_output=print_output)

        cleaned_waveform = reduce_noise(y=noisy_waveform, sr=sr)

        output_path = os.path.join(output_dir, "denoised.wav")

        os.makedirs(output_dir, exist_ok=True)

        sf.write(output_path, cleaned_waveform, sr)

        self.logger.log(f"Denoising completed! Cleaned file: {output_path}", print_output=print_output)

        return output_path


if __name__ == "__main__":
    config = "config/config.yaml"

    test_noisy_audio = ".data/example/LogisticsCallCenterConversation.mp3"
    test_output_dir = ".temp"

    test_denoiser = Denoiser(config_path=config)

    try:
        print(f"Starting denoising for: {test_noisy_audio}")
        result_path = test_denoiser.denoise_audio(
            input_path=test_noisy_audio,
            output_dir=test_output_dir,
            noise_threshold=0.005,
            print_output=True
        )
        if result_path:
            print(f"Denoising completed! Cleaned file saved at: {result_path}")
        else:
            print("Denoising skipped due to low noise level.")
    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")