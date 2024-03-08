# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

os.environ["HF_HOME"] = "/src/hf_models"
os.environ["TORCH_HOME"] = "/src/torch_models"
from cog import BasePredictor, Input, Path, BaseModel
import torch
import whisperx
from typing import Any


compute_type = "float16"


class Output(BaseModel):
    detected_language: str
    transcription: str
    segments: Any
    word_segments: Any


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model(
            "large-v3", self.device, compute_type=compute_type
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        batch_size: int = Input(
            description="Parallelization of input audio transcription",
            ge=1,
            default=32,
        ),
        chunk_size: int = Input(
            description="Chunk size for merging VAD sements",
            ge=1,
            default=30,
        ),
        language: str = Input(description="Language code", default=None),
        align_output: bool = Input(
            description="Use if you need word-level timing and not just batched transcription",
            default=False,
        ),
        debug: bool = Input(description="Print out debug information.", default=False),
    ) -> Output:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            result = self.model.transcribe(
                str(audio),
                batch_size=batch_size,
                chunk_size=chunk_size,
                language=language,
                print_progress=debug,
            )
            transcription = "".join([segment["text"] for segment in result["segments"]])
            if align_output:
                alignment_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=self.device
                )
                aligned_result = whisperx.align(
                    result["segments"],
                    alignment_model,
                    metadata,
                    str(audio),
                    self.device,
                    return_char_alignments=False,
                )
                result.update(aligned_result)
            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB"
                )
        return Output(
            transcription=transcription,
            segments=result["segments"],
            word_segments=result.get("word_segments"),
            detected_language=result["language"],
        )
