from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import bentoml
from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str
    words: list[dict]
    language: str
    processing_time: float


@bentoml.service(
    name="faster_whisper_pl",
    workers=1,
    resources={"gpu": 1},
    traffic={"timeout": 300},
    envs=[{"name": "HF_TOKEN"}],
)
class FasterWhisperService:

    def __init__(self) -> None:
        import site
        import os
        sp = site.getsitepackages()[0]
        cublas_path = os.path.join(sp, "nvidia", "cublas", "lib")
        cudnn_path = os.path.join(sp, "nvidia", "cudnn", "lib")
        os.environ["LD_LIBRARY_PATH"] = cublas_path + ":" + cudnn_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            "large-v3-turbo",
            device="cuda",
            compute_type="float16",
        )
        print("[faster-whisper] Model loaded on cuda")
    

    @bentoml.api(route="/transcribe")
    async def transcribe(
        self,
        upload_file: Path,
        language: Optional[str] = "pl",
        word_timestamps: Optional[bool] = False,
    ) -> dict:
        t_start = time.time()

        segments, info = self.model.transcribe(
            str(upload_file),
            language=language or "pl",
            word_timestamps=word_timestamps,
            vad_filter=True,
        )

        words = []
        full_text = []

        for segment in segments:
            full_text.append(segment.text.strip())
            if word_timestamps and segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                    })

        return TranscriptionResponse(
            text=" ".join(full_text),
            words=words,
            language=info.language,
            processing_time=round(time.time() - t_start, 3),
        ).model_dump()

    @bentoml.api(route="/health")
    async def health(self) -> dict:
        return {"status": "ok"}
