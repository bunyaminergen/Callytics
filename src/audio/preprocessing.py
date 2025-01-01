# Standard library imports
import os
from typing import Annotated

# Related third-party imports
import librosa
import soundfile as sf
from librosa.feature import rms
from omegaconf import OmegaConf
from noisereduce import reduce_noise
from MPSENet import MPSENet

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
    logger : Logger
        Logger instance for recording messages.
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

        Examples
        --------
        >>> denoise = Denoiser("config.yaml")
        >>> input_file = "noisy_audio.wav"
        >>> output_directory = "cleaned_audio"
        >>> noise_thresh = 0.02
        >>> result = denoiser.denoise_audio(input_file, output_directory, noise_thresh)
        >>> print(result)
        cleaned_audio/denoised.wav
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


class SpeechEnhancement:
    """
    A class for speech enhancement using the MPSENet model.

    This class provides methods to load audio, apply enhancement using a
    pre-trained MPSENet model, and save the enhanced output.

    Parameters
    ----------
    config_path : str
        Path to the configuration file specifying runtime settings.
    output_dir : str, optional
        Directory to save enhanced audio files. Defaults to ".temp".

    Attributes
    ----------
    config : omegaconf.DictConfig
        Loaded configuration data.
    output_dir : str
        Directory to save enhanced audio files.
    model_name : str
        Name of the pre-trained model.
    device : str
        Device to run the model (e.g., "cpu" or "cuda").
    model : MPSENet
        Pre-trained MPSENet model instance.
    """

    def __init__(
            self,
            config_path: Annotated[str, "Path to the config file"],
            output_dir: Annotated[str, "Default directory to save enhanced audio files"] = ".temp"
    ) -> None:
        """
        Initialize the SpeechEnhancement class.

        Parameters
        ----------
        config_path : str
            Path to the configuration file specifying runtime settings.
        output_dir : str, optional
            Directory to save enhanced audio files. Defaults to ".temp".
        """
        self.config = OmegaConf.load(config_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_name = self.config.models.mpsenet.model_name
        self.device = self.config.runtime.device

        self.model = MPSENet.from_pretrained(self.model_name).to(self.device)

    def enhance_audio(
            self,
            input_path: Annotated[str, "Path to the original audio file"],
            output_path: Annotated[str, "Path to save the enhanced audio file"],
            noise_threshold: Annotated[float, "Noise threshold value to decide if enhancement is needed"],
            verbose: Annotated[bool, "Whether to log additional info to console"] = False,
    ) -> str:
        """
        Enhance an audio file using the MPSENet model.

        Parameters
        ----------
        input_path : str
            Path to the original input audio file.
        output_path : str
            Path to save the enhanced audio file.
        noise_threshold : float
            Noise threshold value to decide if enhancement is needed.
        verbose : bool, optional
            Whether to log additional info to the console. Defaults to False.

        Returns
        -------
        str
            Path to the enhanced audio file if enhancement is performed, otherwise the original file path.

        Examples
        --------
        >>> enhancer = SpeechEnhancement("config.yaml")
        >>> input_file = "raw_audio.wav"
        >>> output_file = "enhanced_audio.wav"
        >>> noise_thresh = 0.03
        >>> result = enhancer.enhance_audio(input_file, output_file, noise_thresh)
        >>> print(result)
        enhanced_audio.wav
        """
        raw_waveform, sr_raw = librosa.load(input_path, sr=None)
        noise_level = rms(y=raw_waveform).mean()

        if verbose:
            print(f"[SpeechEnhancement] Detected noise level: {noise_level:.6f}")

        if noise_level < noise_threshold:
            if verbose:
                print(f"[SpeechEnhancement] Noise level < {noise_threshold} → enhancement skipped.")
            return input_path

        sr_model = self.model.h.sampling_rate
        waveform, sr = librosa.load(input_path, sr=sr_model)

        if verbose:
            print(f"[SpeechEnhancement] Enhancement with MPSENet started using model: {self.model_name}")

        enhanced_waveform, sr_out, _ = self.model(waveform)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, enhanced_waveform, sr_out)

        if verbose:
            print(f"[SpeechEnhancement] Enhancement complete. Saved to: {output_path}")

        return output_path


if __name__ == "__main__":

    test_config_path = "config/config.yaml"
    noisy_audio_file = ".data/example/noisy/LookOncetoHearTargetSpeechHearingwithNoisyExamples.mp3"
    temp_dir = ".temp"

    denoiser = Denoiser(config_path=test_config_path, output_dir=temp_dir)
    denoised_path = denoiser.denoise_audio(
        input_path=noisy_audio_file,
        output_dir=temp_dir,
        noise_threshold=0.005,
        print_output=True
    )
    if denoised_path == noisy_audio_file:
        print("Denoising skipped due to low noise level.")
    else:
        print(f"Denoising completed! Cleaned file saved at: {denoised_path}")

    speech_enhancer = SpeechEnhancement(config_path=test_config_path, output_dir=temp_dir)
    enhanced_audio_path = os.path.join(temp_dir, "enhanced_audio.wav")

    result_path = speech_enhancer.enhance_audio(
        input_path=denoised_path,
        output_path=enhanced_audio_path,
        noise_threshold=0.005,
        verbose=True
    )

    if result_path == denoised_path:
        print("Enhancement skipped due to low noise level.")
    else:
        print(f"Speech enhancement completed! Enhanced file saved at: {result_path}")
