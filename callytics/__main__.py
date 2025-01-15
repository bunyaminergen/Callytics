# Standard library imports
import os

# Related third-party imports
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

# Local imports
from callytics.audio.utils import Formatter
from callytics.audio.metrics import SilenceStats
from callytics.audio.error import DialogueDetecting
from callytics.audio.alignment import ForcedAligner
from callytics.audio.effect import DemucsVocalSeparator
from callytics.audio.preprocessing import SpeechEnhancement
from callytics.audio.io import SpeakerTimestampReader, TranscriptWriter
from callytics.audio.analysis import WordSpeakerMapper, SentenceSpeakerMapper, Audio
from callytics.audio.processing import AudioProcessor, Transcriber, PunctuationRestorer
from callytics.text.utils import Annotator
from callytics.text.llm import LLMOrchestrator, LLMResultHandler
from callytics.utils.utils import Cleaner, Watcher
from callytics.db.manager import Database


async def main(audio_file_path: str):
    """
    Process an audio file to perform diarization, transcription, punctuation restoration,
    and speaker role classification.

    Parameters
    ----------
    audio_file_path : str
        The path to the input audio file to be processed.

    Returns
    -------
    None
    """
    # Paths
    config_nemo = "config/nemo/diar_infer_telephonic.yaml"
    manifest_path = ".temp/manifest.json"
    temp_dir = ".temp"
    rttm_file_path = os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")
    transcript_output_path = ".temp/output.txt"
    srt_output_path = ".temp/output.srt"
    config_path = "config/config.yaml"
    prompt_path = "config/prompt.yaml"
    db_path = ".db/Callytics.sqlite"
    db_topic_fetch_path = "src/db/sql/TopicFetch.sql"
    db_topic_insert_path = "src/db/sql/TopicInsert.sql"
    db_audio_properties_insert_path = "src/db/sql/AudioPropertiesInsert.sql"
    db_utterance_insert_path = "src/db/sql/UtteranceInsert.sql"

    # Configuration
    config = OmegaConf.load(config_path)
    device = config.runtime.device
    compute_type = config.runtime.compute_type
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.runtime.cuda_alloc_conf

    # Initialize Classes
    dialogue_detector = DialogueDetecting(delete_original=True)
    enhancer = SpeechEnhancement(config_path=config_path, output_dir=temp_dir)
    separator = DemucsVocalSeparator()
    processor = AudioProcessor(audio_path=audio_file_path, temp_dir=temp_dir)
    transcriber = Transcriber(device=device, compute_type=compute_type)
    aligner = ForcedAligner(device=device)
    llm_handler = LLMOrchestrator(config_path=config_path, prompt_config_path=prompt_path, model_id="openai")
    llm_result_handler = LLMResultHandler()
    cleaner = Cleaner()
    formatter = Formatter()
    db = Database(db_path)
    audio_feature_extractor = Audio(audio_file_path)

    # Step 1: Detect Dialogue
    has_dialogue = dialogue_detector.process(audio_file_path)
    if not has_dialogue:
        return

    # Step 2: Speech Enhancement
    audio_path = enhancer.enhance_audio(
        input_path=audio_file_path,
        output_path=os.path.join(temp_dir, "enhanced.wav"),
        noise_threshold=0.0001,
        verbose=True
    )

    # Step 3: Vocal Separation
    vocal_path = separator.separate_vocals(audio_file=audio_path, output_dir=temp_dir)

    # Step 4: Transcription
    transcript, info = transcriber.transcribe(audio_path=vocal_path)
    detected_language = info["language"]

    # Step 5: Forced Alignment
    word_timestamps = aligner.align(
        audio_path=vocal_path,
        transcript=transcript,
        language=detected_language
    )

    # Step 6: Diarization
    processor.audio_path = vocal_path
    mono_audio_path = processor.convert_to_mono()
    processor.audio_path = mono_audio_path
    processor.create_manifest(manifest_path)
    cfg = OmegaConf.load(config_nemo)
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = temp_dir
    msdd_model = NeuralDiarizer(cfg=cfg)
    msdd_model.diarize()

    # Step 7: Processing Transcript
    # Step 7.1: Speaker Timestamps
    speaker_reader = SpeakerTimestampReader(rttm_path=rttm_file_path)
    speaker_ts = speaker_reader.read_speaker_timestamps()

    # Step 7.2: Mapping Words
    word_speaker_mapper = WordSpeakerMapper(word_timestamps, speaker_ts)
    wsm = word_speaker_mapper.get_words_speaker_mapping()

    # Step 7.3: Punctuation Restoration
    punct_restorer = PunctuationRestorer(language=detected_language)
    wsm = punct_restorer.restore_punctuation(wsm)
    word_speaker_mapper.word_speaker_mapping = wsm
    word_speaker_mapper.realign_with_punctuation()
    wsm = word_speaker_mapper.word_speaker_mapping

    # Step 7.4: Mapping Sentences
    sentence_mapper = SentenceSpeakerMapper()
    ssm = sentence_mapper.get_sentences_speaker_mapping(wsm)

    # Step 8 (Optional): Write Transcript and SRT Files
    writer = TranscriptWriter()
    writer.write_transcript(ssm, transcript_output_path)
    writer.write_srt(ssm, srt_output_path)

    # Step 9: Classify Speaker Roles
    speaker_roles = await llm_handler.generate("Classification", ssm)

    # Step 9.1: LLM results validate and fallback
    ssm = llm_result_handler.validate_and_fallback(speaker_roles, ssm)
    llm_result_handler.log_result(ssm, speaker_roles)

    # Step 10: Sentiment Analysis
    ssm_with_indices = formatter.add_indices_to_ssm(ssm)
    annotator = Annotator(ssm_with_indices)
    sentiment_results = await llm_handler.generate("SentimentAnalysis", user_input=ssm)
    annotator.add_sentiment(sentiment_results)

    # Step 11: Profanity Word Detection
    profane_results = await llm_handler.generate("ProfanityWordDetection", user_input=ssm)
    annotator.add_profanity(profane_results)

    # Step 12: Summary
    summary_result = await llm_handler.generate("Summary", user_input=ssm)
    annotator.add_summary(summary_result)

    # Step 13: Conflict Detection
    conflict_result = await llm_handler.generate("ConflictDetection", user_input=ssm)
    annotator.add_conflict(conflict_result)

    # Step 14: Topic Detection
    topics = db.fetch(db_topic_fetch_path)
    topic_result = await llm_handler.generate(
        "TopicDetection",
        user_input=ssm,
        system_input=topics
    )
    annotator.add_topic(topic_result)

    #  Step 15: File/Audio Feature Extraction
    props = audio_feature_extractor.properties()

    (
        name,
        file_extension,
        absolute_file_path,
        sample_rate,
        min_frequency,
        max_frequency,
        audio_bit_depth,
        num_channels,
        audio_duration,
        rms_loudness,
        final_features
    ) = props

    rms_loudness_db = final_features["RMSLoudness"]
    zero_crossing_rate_db = final_features["ZeroCrossingRate"]
    spectral_centroid_db = final_features["SpectralCentroid"]
    eq_20_250_db = final_features["EQ_20_250_Hz"]
    eq_250_2000_db = final_features["EQ_250_2000_Hz"]
    eq_2000_6000_db = final_features["EQ_2000_6000_Hz"]
    eq_6000_20000_db = final_features["EQ_6000_20000_Hz"]
    mfcc_values = [final_features[f"MFCC_{i}"] for i in range(1, 14)]

    final_output = annotator.finalize()

    # Step 16: Tocal Silence Calculation
    stats = SilenceStats.from_segments(final_output['ssm'])
    t_std = stats.threshold_std(factor=0.99)
    final_output["silence"] = t_std

    print("Final_Output:", final_output)

    # Step 17: Database
    # Step 17.1: Insert File Table
    summary = final_output.get("summary", "")
    conflict_flag = 1 if final_output.get("conflict", False) else 0
    silence_value = final_output.get("silence", 0.0)
    detected_topic = final_output.get("topic", "Unknown")

    topic_id = db.get_or_insert_topic_id(detected_topic, topics, db_topic_insert_path)

    params = (
        name,
        topic_id,
        file_extension,
        absolute_file_path,
        sample_rate,
        min_frequency,
        max_frequency,
        audio_bit_depth,
        num_channels,
        audio_duration,
        rms_loudness_db,
        zero_crossing_rate_db,
        spectral_centroid_db,
        eq_20_250_db,
        eq_250_2000_db,
        eq_2000_6000_db,
        eq_6000_20000_db,
        *mfcc_values,
        summary,
        conflict_flag,
        silence_value
    )

    last_id = db.insert(db_audio_properties_insert_path, params)
    print(f"Audio properties inserted successfully into the File table with ID: {last_id}")

    # Step 17.2: Insert Utterance Table
    utterances = final_output["ssm"]

    for utterance in utterances:
        file_id = last_id
        speaker = utterance["speaker"]
        sequence = utterance["index"]
        start_time = utterance["start_time"] / 1000.0
        end_time = utterance["end_time"] / 1000.0
        content = utterance["text"]
        sentiment = utterance["sentiment"]
        profane = 1 if utterance["profane"] else 0

        utterance_params = (
            file_id,
            speaker,
            sequence,
            start_time,
            end_time,
            content,
            sentiment,
            profane
        )

        db.insert(db_utterance_insert_path, utterance_params)

    print("Utterances inserted successfully into the Utterance table.")

    # Step 18: Clean Up
    cleaner.cleanup(temp_dir, audio_file_path)


async def process(path: str):
    """
    Asynchronous callback function that is triggered when a new audio file is detected.

    Parameters
    ----------
    path : str
        The path to the newly created audio file.

    Returns
    -------
    None
    """
    print(f"Processing new audio file: {path}")
    await main(path)


if __name__ == "__main__":
    directory_to_watch = ".data/input"
    Watcher.start_watcher(directory_to_watch, process)
