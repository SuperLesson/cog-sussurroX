import os

os.environ["HF_HOME"] = "/src/hf_models"
os.environ["TORCH_HOME"] = "/src/torch_models"

import gc
import subprocess
import tempfile
import typing as t
from datetime import datetime, time, timedelta
from time import time as now

import torch
import whisperx
from cog import BaseModel, BasePredictor, Input, Path
from whisperx.audio import N_SAMPLES, log_mel_spectrogram


def from_seconds(seconds: float) -> time:
    """Convert seconds to time."""
    assert seconds >= 0
    return (datetime.min + timedelta(seconds=seconds)).time()


def time_it(fmt: str):
    """Decorator to time a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            _start = now()
            result = func(*args, **kwargs)
            _end = now()
            print(fmt.format(duration=from_seconds(_end - _start)))
            return result

        return wrapper

    return decorator


LAST_SETTINGS = {
    "vad_onset": 0.5,
    "vad_offset": 0.363,
    "diarization": False,
    # "initial_prompt": None,
    "temperatures": [0],
    "align_output": True,
    "language": "en",
    "language_detection_min_prob": 0,
    "language_detection_max_tries": 5,
}


def check_outdated(settings: dict) -> bool:
    global LAST_SETTINGS
    if any(LAST_SETTINGS.get(k) != v for k, v in settings.items()):
        LAST_SETTINGS.update(settings)
        return True
    return False


WHISPER_ARCH = "large-v3"
# change to "int8" if low on GPU mem (may reduce accuracy)
COMPUTE_TYPE = "float16"
DEVICE = "cuda"
MIN_SEG_LEN = 30


WHISPER_MODEL: t.Any = None
ALIGN_MODEL: t.Any = None
ALIGN_META: t.Any = None
DIARIZE_MODEL: t.Any = None


class Output(BaseModel):
    segments: t.Any
    detected_language: str


@time_it("Took {duration} to load model")
def load_audio_model(arch, language, asr_options, vad_options):
    global WHISPER_MODEL
    WHISPER_MODEL = whisperx.load_model(
        arch,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=language,
        asr_options=asr_options,
        vad_options=vad_options,
    )


class Predictor(BasePredictor):
    def setup(self):
        global ALIGN_MODEL, ALIGN_META
        asr_options = {
            "temperatures": [0],
            "initial_prompt": None,
        }

        vad_options = {"vad_onset": 0.5, "vad_offset": 0.363}
        load_audio_model(WHISPER_ARCH, "en", asr_options, vad_options)

        ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(
            language_code="en", device=DEVICE
        )

    def predict(
        self,
        audio_file: Path = Input(description="Audio file"),
        language: str = Input(
            description="ISO code of the language spoken in the audio, specify None to perform language detection",
            default=None,
        ),
        language_detection_min_prob: float = Input(
            description="If language is not specified, then the language will be detected recursively on different "
            "parts of the file until it reaches the given probability",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="If language is not specified, then the language will be detected following the logic of "
            "language_detection_min_prob parameter, but will stop after the given max retries. If max "
            "retries is reached, the most probable language is kept.",
            default=5,
        ),
        initial_prompt: str = Input(
            description="Optional text to provide as a prompt for the first window",
            default=None,
        ),
        batch_size: int = Input(
            description="Parallelization of input audio transcription", default=64
        ),
        temperature: float = Input(
            description="Temperature to use for sampling", default=0
        ),
        vad_onset: float = Input(description="VAD onset", default=0.500),
        vad_offset: float = Input(description="VAD offset", default=0.363),
        diarization: bool = Input(
            description="Assign speaker ID labels", default=False
        ),
        huggingface_access_token: str = Input(
            description="To enable diarization, please enter your HuggingFace token (read). You need to accept "
            "the user agreement for the models specified in the README.",
            default=None,
        ),
        min_speakers: int = Input(
            description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
            default=None,
        ),
        debug: bool = Input(
            description="Print out compute/inference times and memory usage information",
            default=False,
        ),
        distil: bool = Input(description="Use distilled Whisper model", default=False),
    ) -> Output:
        with torch.inference_mode():
            asr_options = {
                "temperatures": [temperature],
                "initial_prompt": initial_prompt,
            }

            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

            if distil:
                if language and language != "en":
                    msg = "Distilled model only supports English"
                    raise Exception(msg)

                arch = f"distil-{WHISPER_ARCH}"
                load_audio_model(arch, "en", asr_options, vad_options)
            else:
                arch = WHISPER_ARCH

            if not language:

                @time_it("Language detection took {duration}")
                def detect_language() -> str:
                    duration = (
                        float(
                            subprocess.run(
                                [
                                    "ffprobe",
                                    "-i",
                                    audio_file,
                                    "-show_entries",
                                    "format=duration",
                                    "-v",
                                    "quiet",
                                    "-of",
                                    "csv=p=0",
                                ],
                                capture_output=True,
                                text=True,
                            ).stdout.strip()
                        )
                        / 1000
                    )

                    dumb_model = whisperx.load_model(
                        arch,
                        DEVICE,
                        compute_type=COMPUTE_TYPE,
                    )

                    best_guess = ""
                    best_prob = 0

                    for i in range(
                        min(
                            language_detection_max_tries,
                            int(duration / MIN_SEG_LEN),
                        )
                    ):
                        if language_detection_min_prob < 0.01:
                            msg = "Cannot guess language based on nothing"
                            raise Exception(msg)

                        start = i * MIN_SEG_LEN

                        print(
                            f"Detecting language using {from_seconds(start)} - {from_seconds(start + MIN_SEG_LEN)}"
                        )

                        audio_slice = Path(
                            tempfile.NamedTemporaryFile(
                                suffix=".wav", delete=False
                            ).name
                        )

                        subprocess.run(
                            [  # noqa: S607
                                "ffmpeg",
                                "-loglevel",
                                "quiet",
                                "-i",
                                audio_file,
                                "-vn",
                                "-y",
                                "-acodec",
                                "pcm_s16le",
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                "-ss",
                                str(start),
                                "-t",
                                str(MIN_SEG_LEN),
                                audio_slice,
                            ],
                            stdout=subprocess.DEVNULL,
                        )

                        audio = whisperx.load_audio(audio_slice)

                        model_n_mels = dumb_model.model.feat_kwargs.get("feature_size")
                        segment = log_mel_spectrogram(
                            audio[:N_SAMPLES],
                            n_mels=model_n_mels if model_n_mels is not None else 80,
                            padding=0
                            if audio.shape[0] >= N_SAMPLES
                            else N_SAMPLES - audio.shape[0],
                        )
                        encoder_output = dumb_model.model.encode(segment)
                        results = dumb_model.model.model.detect_language(encoder_output)
                        language_token, language_probability = results[0][0]
                        detected_language = language_token[2:-2]

                        print(
                            f"Iteration {i} - Detected language: {detected_language} ({language_probability:.2f})"
                        )

                        audio_slice.unlink()
                        gc.collect()
                        torch.cuda.empty_cache()

                        if language_probability > best_prob:
                            best_guess = detected_language

                        if (
                            language_probability >= language_detection_min_prob
                            or i >= language_detection_max_tries
                        ):
                            del dumb_model
                            return detected_language
                    return best_guess

                language = detect_language()

            settings = {"language": language, **asr_options, **vad_options}

            if check_outdated(settings):
                load_audio_model(arch, language, asr_options, vad_options)

            @time_it("Took {duration} to load audio")
            def load_audio():
                return whisperx.load_audio(audio_file)

            audio = load_audio()

            @time_it("Took {duration} to transcribe")
            def transcribe_audio():
                return WHISPER_MODEL.transcribe(audio, batch_size=batch_size)

            result = transcribe_audio()

            detected_language = result["language"]

            gc.collect()
            torch.cuda.empty_cache()

            if (
                detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
                or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
            ):
                result = align(audio, detected_language, result["segments"])
            else:
                print(
                    f"Cannot align output as language {detected_language} is not supported for alignment"
                )

            if diarization:
                result = diarize(
                    audio,
                    result,
                    huggingface_access_token,
                    min_speakers,
                    max_speakers,
                )

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )

        return Output(segments=result["segments"], detected_language=detected_language)


@time_it("Took {duration} to align output")
def align(audio, language, segments):
    if check_outdated({"language": language}):
        global ALIGN_MODEL, ALIGN_META
        ALIGN_MODEL, ALIGN_META = whisperx.load_align_model(
            language_code=language, device=DEVICE
        )

    result = whisperx.align(
        segments,
        ALIGN_MODEL,
        ALIGN_META,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return result


@time_it("Duration to diarize segments: {duration}")
def diarize(audio, result, huggingface_access_token, min_speakers, max_speakers):
    global DIARIZE_MODEL
    DIARIZE_MODEL = whisperx.DiarizationPipeline(
        model_name="pyannote/speaker-diarization@2.1",
        use_auth_token=huggingface_access_token,
        device=DEVICE,
    )
    diarize_segments = DIARIZE_MODEL(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    gc.collect()
    torch.cuda.empty_cache()
    del DIARIZE_MODEL

    return result
