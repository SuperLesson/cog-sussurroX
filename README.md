# Cog-sussurroX

Cog predictor interface implementation of [WhisperX](https://github.com/m-bain/whisperX).

The cog model is available as [isinyaaa/whisperx](https://replicate.com/isinyaaa/whisperx).

## Running locally

1. Install the [cog CLI](https://github.com/replicate/cog).
2. Clone this repository
3. Run

    ```sh
    cog predict -i audio=@/path/to/audio/file [-i other_flag=val ...]
    ```

You can check out the available flags through [Replicate](https://replicate.com/isinyaaa/whisperx).

## Model Information

WhisperX provides fast automatic speech recognition (70x realtime with large-v3) with word-level timestamps and speaker diarization.

Whisper is an ASR model developed by OpenAI, trained on a large dataset of diverse audio. Whilst it does produces highly accurate transcriptions, the corresponding timestamps are at the utterance-level, not per word, and can be inaccurate by several seconds. OpenAI’s whisper does not natively support batching, but WhisperX does.

Model used is for transcription is large-v3 from faster-whisper.

For more information about WhisperX, including implementation details, see the [WhisperX github repo](https://github.com/m-bain/whisperX).

## Citation

```tex
@misc{bain2023whisperx,
      title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio}, 
      author={Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
      year={2023},
      eprint={2303.00747},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
