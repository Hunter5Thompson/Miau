# services/transcription_service.py
"""
Transcription Service

Kapselt die komplette WhisperX-Transkriptionslogik in einen sauberen Service.
Wie ein professioneller Übersetzer, der verschiedene Sprachen und Dialekte
beherrscht und genau weiß, welche Technik für welchen Text am besten funktioniert.
"""

import asyncio
import logging
import time
import gc
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, replace as dataclass_replace # Importiere replace
from enum import Enum
import numpy as np

import torch
from pydantic import ConfigDict # Für Pydantic v2 Namespace-Konflikte

from config.model_config import ModelLoader, ModelType, ModelTypeDetector
from config.hardware_detection import HardwareCapabilities

logger = logging.getLogger(__name__)


class TranscriptionQuality(Enum):
    """Transcription quality levels"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    PREMIUM = "premium"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription process"""
    language: Optional[str] = "de"
    quality: TranscriptionQuality = TranscriptionQuality.BALANCED
    beam_size: int = 3
    word_timestamps: bool = True
    condition_on_previous_text: bool = False
    temperature: Union[float, List[float], Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    prepend_punctuations: str = "\"'¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：)]}、"
    batch_size: Optional[int] = None
    use_vad_filter: bool = False
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_max_speech_duration_s: float = float("inf")
    vad_min_silence_duration_ms: int = 2000
    vad_speech_pad_ms: int = 400

    @classmethod
    def get_fast_config(cls) -> 'TranscriptionConfig':
        return cls(
            language="de", quality=TranscriptionQuality.FAST, beam_size=1,
            word_timestamps=False, temperature=0.0, use_vad_filter=True,
            vad_threshold=0.5, vad_min_speech_duration_ms=500,
            vad_min_silence_duration_ms=2500, vad_speech_pad_ms=100
        )
    @classmethod
    def get_balanced_config(cls) -> 'TranscriptionConfig':
        return cls(
            language="de", quality=TranscriptionQuality.BALANCED, beam_size=3,
            word_timestamps=True, temperature=(0.0, 0.2, 0.4), use_vad_filter=True,
            vad_threshold=0.5, vad_min_speech_duration_ms=250,
            vad_min_silence_duration_ms=2000, vad_speech_pad_ms=400
        )
    @classmethod
    def get_accurate_config(cls) -> 'TranscriptionConfig':
        return cls(
            language="de", quality=TranscriptionQuality.ACCURATE, beam_size=5,
            word_timestamps=True, temperature=(0.0, 0.2, 0.4, 0.6), use_vad_filter=True,
            vad_threshold=0.4, vad_min_speech_duration_ms=150,
            vad_min_silence_duration_ms=1500, vad_speech_pad_ms=300,
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6
        )
    @classmethod
    def get_premium_config(cls, hardware_caps: HardwareCapabilities) -> 'TranscriptionConfig':
        config = cls(
            language="de", quality=TranscriptionQuality.PREMIUM, beam_size=5,
            word_timestamps=True, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            use_vad_filter=True, vad_threshold=0.35, vad_min_speech_duration_ms=100,
            vad_min_silence_duration_ms=1000, vad_speech_pad_ms=200,
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
            prepend_punctuations="\"'¿([{-", append_punctuations="\"'.。,，!！?？:：)]}、"
        )
        if (hardware_caps.gpu_name and
                any(gpu in hardware_caps.gpu_name for gpu in ["H100", "A100", "L40S", "RTX_4090"])):
            config.beam_size = 7
        return config

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None
    def duration(self) -> float: return self.end - self.start
    def word_count(self) -> int: return len(self.text.split()) if self.text else 0

@dataclass
class TranscriptionResult:
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    model_name: str
    config: TranscriptionConfig
    processing_time: float
    model_load_time: Optional[float] = None
    transcription_time: Optional[float] = None
    average_confidence: Optional[float] = None
    segments_count: int = 0
    words_count: int = 0

    # Pydantic v2 model_config zur Behandlung von Namespace-Konflikten
    # für Felder wie "model_name"
    model_config = ConfigDict(protected_namespaces=())


    def __post_init__(self):
        self.segments_count = len(self.segments)
        self.words_count = sum(seg.word_count() for seg in self.segments)
        confidences = [seg.confidence for seg in self.segments if seg.confidence is not None and isinstance(seg.confidence, (float, int))]
        if confidences:
            self.average_confidence = sum(confidences) / len(confidences)

    def get_full_text(self, separator: str = " ") -> str:
        return separator.join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "duration_seconds": self.duration,
            "processing_time_seconds": self.processing_time,
            "model_load_time_seconds": self.model_load_time,
            "transcription_engine_time_seconds": self.transcription_time,
            "segments_count": self.segments_count,
            "words_count": self.words_count,
            "words_per_minute": (self.words_count / (self.duration / 60)) if self.duration > 0 else 0,
            "average_segment_confidence": self.average_confidence,
            "language_detected": self.language,
            "model_name_used": self.model_name,
            "quality_level_config": self.config.quality.value,
            "beam_size_config": self.config.beam_size,
            "vad_used_config": self.config.use_vad_filter
        }

class TranscriptionEngine:
    def transcribe(self, audio: np.ndarray, model: Any, config: TranscriptionConfig, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    def get_engine_name(self) -> str:
        raise NotImplementedError

class FasterWhisperEngine(TranscriptionEngine):
    def transcribe(self, audio: np.ndarray, model: Any, config: TranscriptionConfig, **kwargs) -> Dict[str, Any]:
        logger.info(f"🎯 Transcribing with faster-whisper (quality: {config.quality.value}, lang: {config.language})")

        # KORREKTUR: Sicherstellen, dass Audio np.float32 und C-kontinuierlich ist
        if audio.dtype != np.float32:
            logger.debug(f"Audio data type is {audio.dtype}, converting to np.float32 for faster-whisper.")
            audio = audio.astype(np.float32)
        if not audio.flags['C_CONTIGUOUS']:
            logger.debug("Audio data is not C-contiguous for faster-whisper, making a contiguous copy.")
            audio = np.ascontiguousarray(audio)

        params = {
            "language": config.language if config.language and config.language.strip() else None,
            "beam_size": config.beam_size,
            "word_timestamps": config.word_timestamps,
            "condition_on_previous_text": config.condition_on_previous_text,
            "temperature": config.temperature,
            "compression_ratio_threshold": config.compression_ratio_threshold,
            "log_prob_threshold": config.log_prob_threshold,
            "no_speech_threshold": config.no_speech_threshold,
            "prepend_punctuations": config.prepend_punctuations,
            "append_punctuations": config.append_punctuations,
        }
        if config.use_vad_filter:
            params["vad_filter"] = True
            params["vad_parameters"] = {
                "threshold": config.vad_threshold,
                "min_speech_duration_ms": config.vad_min_speech_duration_ms,
                "max_speech_duration_s": config.vad_max_speech_duration_s,
                "min_silence_duration_ms": config.vad_min_silence_duration_ms,
                "speech_pad_ms": config.vad_speech_pad_ms
            }
            logger.debug(f"VAD parameters for faster-whisper: {params['vad_parameters']}")
        else:
            params["vad_filter"] = False

        segments_iter, info = model.transcribe(audio, **params)
        segments_list = []
        for segment in segments_iter:
            segment_dict = {
                "start": float(segment.start), "end": float(segment.end),
                "text": segment.text.strip(), "confidence": getattr(segment, 'avg_logprob', None)
            }
            if config.word_timestamps and hasattr(segment, 'words'):
                segment_dict["words"] = [
                    {"start": float(word.start), "end": float(word.end), "word": word.word, "probability": float(word.probability)}
                    for word in segment.words if hasattr(word, 'start')
                ]
            segments_list.append(segment_dict)

        detected_language = getattr(info, 'language', config.language or "unknown")
        language_probability = getattr(info, 'language_probability', None) # Kann float, int oder None sein
        processing_duration_info = getattr(info, 'duration', None) # Kann float, int oder None sein

        # KORREKTUR HIER: Bereite die Strings für die Log-Ausgabe separat vor
        prob_text = "N/A"
        if language_probability is not None:
            try:
                # Konvertiere zu float, falls es ein int ist, um :.2f sicher anzuwenden
                prob_text = f"{float(language_probability):.2f}"
            except (ValueError, TypeError):
                logger.warning(f"Konnte language_probability ('{language_probability}') nicht als float formatieren.")
                prob_text = str(language_probability) # Fallback auf String-Repräsentation

        duration_text = "N/A"
        if processing_duration_info is not None:
            try:
                # Konvertiere zu float, falls es ein int ist
                duration_text = f"{float(processing_duration_info):.2f}s"
            except (ValueError, TypeError):
                logger.warning(f"Konnte processing_duration_info ('{processing_duration_info}') nicht als float formatieren.")
                duration_text = f"{processing_duration_info}s" # Fallback

        logger.info(
            f"✅ faster-whisper transcription complete: {len(segments_list)} segments. "
            f"Detected language: {detected_language} (Prob: {prob_text}) "
            f"Audio duration processed by model: {duration_text}"
        )
        # Gib die Originalwerte zurück, nicht die formatierten Strings
        return {"segments": segments_list, "language": detected_language, "language_probability": language_probability}

    def get_engine_name(self) -> str: return "faster-whisper"

class WhisperXEngine(TranscriptionEngine):
    def transcribe(self, audio: np.ndarray, model: Any, config: TranscriptionConfig, **kwargs) -> Dict[str, Any]:
        logger.info(f"🎯 Transcribing with WhisperX ASR (quality: {config.quality.value}, lang: {config.language})")
        
        # KORREKTUR: Sicherstellen, dass Audio np.float32 und C-kontinuierlich ist
        if audio.dtype != np.float32:
            logger.debug(f"Audio data type is {audio.dtype}, converting to np.float32 for WhisperX ASR.")
            audio = audio.astype(np.float32)
        if not audio.flags['C_CONTIGUOUS']:
            logger.debug("Audio data is not C-contiguous for WhisperX ASR, making a contiguous copy.")
            audio = np.ascontiguousarray(audio)
            
        batch_size = config.batch_size or kwargs.get('batch_size', 16)
        result = model.transcribe(
            audio, batch_size=batch_size,
            language=config.language if config.language and config.language.strip() else None,
            chunk_size=kwargs.get("chunk_size", 30),
            print_progress=kwargs.get("print_progress", False)
        )
        processed_segments = []
        if "segments" in result and isinstance(result["segments"], list):
            for seg_data in result["segments"]:
                processed_segments.append({
                    "start": float(seg_data.get("start", 0.0)), "end": float(seg_data.get("end", 0.0)),
                    "text": seg_data.get("text", "").strip(), "confidence": None,
                    "words": seg_data.get("words")
                })
        detected_language = result.get("language", config.language or "unknown")
        logger.info(f"✅ WhisperX ASR transcription complete: {len(processed_segments)} segments. Detected language: {detected_language}")
        return {"segments": processed_segments, "language": detected_language}
    def get_engine_name(self) -> str: return "whisperx-asr"

class TranscriptionService:
    def __init__(self, model_loader: ModelLoader, hardware_caps: HardwareCapabilities):
        self.model_loader = model_loader
        self.hardware_caps = hardware_caps
        self.engines = {
            ModelType.FASTER_WHISPER: FasterWhisperEngine(),
            ModelType.WHISPERX: WhisperXEngine(),
        }
        self.default_configs = {
            quality: getattr(TranscriptionConfig, f"get_{quality.value}_config")()
                       if quality != TranscriptionQuality.PREMIUM else TranscriptionConfig.get_premium_config(hardware_caps)
            for quality in TranscriptionQuality
        }

    async def transcribe_audio(
        self, audio: np.ndarray, sample_rate: int = 16000, model_name: str = "large-v2",
        config: Optional[TranscriptionConfig] = None, progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> TranscriptionResult:
        effective_config = config or self.default_configs.get(TranscriptionQuality.BALANCED)
        if not effective_config: effective_config = TranscriptionConfig.get_balanced_config()
        overall_start_time = time.time()
        logger.info(
            f"🚀 Starting transcription: {len(audio)/sample_rate:.2f}s audio, "
            f"model={model_name}, quality={effective_config.quality.value}, lang={effective_config.language}"
        )
        if progress_callback: progress_callback("loading_model", 0.1)
        model_obj = None
        # cache_key wird nicht mehr für die Logik im finally-Block benötigt, da is_model_cached entfernt wurde

        try:
            model_load_start_time = time.time()
            model_obj = await asyncio.to_thread(
                self.model_loader.load_transcription_model, model_name,
                self.hardware_caps.device, self.hardware_caps.compute_type
            )
            model_load_duration = time.time() - model_load_start_time
            logger.info(f"✅ Model '{model_name}' loaded in {model_load_duration:.2f}s")
            if progress_callback: progress_callback("model_loaded_starting_transcription", 0.3)

            model_type = ModelTypeDetector.detect_type(model_obj)
            engine = self.engines.get(model_type)
            if not engine:
                raise ValueError(f"No transcription engine for model type: {model_type} (model: {model_name})")
            logger.info(f"🧠 Using engine: {engine.get_engine_name()} for model type: {model_type}")

            final_config = self._adjust_config_for_hardware(effective_config)
            logger.debug(f"Using final transcription config: {final_config}")

            transcription_engine_start_time = time.time()
            def _transcribe_task():
                return engine.transcribe(
                    audio=audio, model=model_obj, config=final_config,
                    batch_size=self.hardware_caps.max_batch_size
                )
            raw_engine_result = await asyncio.to_thread(_transcribe_task)
            transcription_engine_duration = time.time() - transcription_engine_start_time
            logger.info(f"🎤 Engine transcription finished in {transcription_engine_duration:.2f}s")
            if progress_callback: progress_callback("processing_engine_results", 0.8)

            processed_segments = self._process_raw_segments(raw_engine_result.get("segments", []))
            detected_language = raw_engine_result.get("language", final_config.language or "unknown")
            total_service_time = time.time() - overall_start_time
            audio_duration_secs = len(audio) / sample_rate

            transcription_result_obj = TranscriptionResult(
                segments=processed_segments, language=detected_language, duration=audio_duration_secs,
                model_name=model_name, config=final_config, processing_time=total_service_time,
                model_load_time=model_load_duration, transcription_time=transcription_engine_duration
            )
            if progress_callback: progress_callback("completed", 1.0)
            logger.info(
                f"✅ Transcription service finished: {transcription_result_obj.segments_count} segments, "
                f"{transcription_result_obj.words_count} words in {total_service_time:.2f}s. "
                f"(Speed: {audio_duration_secs/total_service_time:.1f}x realtime if total_service_time > 0 else 'N/A')"
            )
            return transcription_result_obj
        except Exception as e:
            logger.error(f"❌ Transcription service failed for model {model_name}: {e}", exc_info=True)
            if progress_callback: progress_callback("error", 1.0)
            raise
        finally:
            # KORREKTUR: Der finally-Block wurde vereinfacht und ruft keine nicht-existente Methode auf.
            if model_obj is not None:
                logger.debug(f"Cleaning up local reference to model {model_name} in TranscriptionService.")
                try:
                    del model_obj # Gibt die lokale Referenz frei
                    gc.collect()
                    if self.hardware_caps.device == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e_cleanup:
                    logger.warning(f"⚠️ Model cleanup warning in TranscriptionService: {e_cleanup}")

    def _adjust_config_for_hardware(self, config: TranscriptionConfig) -> TranscriptionConfig:
        adjusted_config = dataclass_replace(config) # Verwende importiertes replace
        if adjusted_config.batch_size is None:
            adjusted_config.batch_size = self.hardware_caps.max_batch_size
            logger.debug(f"Adjusted batch_size to hardware_caps: {adjusted_config.batch_size}")
        if (self.hardware_caps.device == "cpu" or
                (self.hardware_caps.gpu_memory_gb and self.hardware_caps.gpu_memory_gb < 8)):
            logger.info("⚡ Adjusting transcription config for limited hardware (CPU or <8GB GPU VRAM)")
            adjusted_config.beam_size = min(adjusted_config.beam_size, 2 if self.hardware_caps.device == "cpu" else 3)
            current_batch_size = adjusted_config.batch_size if adjusted_config.batch_size is not None else 16
            adjusted_config.batch_size = min(current_batch_size, 4 if self.hardware_caps.device == "cpu" else 8)
            if isinstance(adjusted_config.temperature, (list, tuple)):
                adjusted_config.temperature = adjusted_config.temperature[0]
        logger.debug(f"Final adjusted config for transcription: beam_size={adjusted_config.beam_size}, batch_size={adjusted_config.batch_size}")
        return adjusted_config

    def _process_raw_segments(self, raw_segments_list: List[Dict[str, Any]]) -> List[TranscriptionSegment]:
        processed_segments = []
        for raw_seg_dict in raw_segments_list:
            if not isinstance(raw_seg_dict, dict):
                logger.warning(f"Skipping invalid raw segment (not a dict): {raw_seg_dict}")
                continue
            text = str(raw_seg_dict.get("text", "")).strip()
            if not text: continue
            segment = TranscriptionSegment(
                start=float(raw_seg_dict.get("start", 0.0)), end=float(raw_seg_dict.get("end", 0.0)),
                text=text, confidence=raw_seg_dict.get("confidence"), words=raw_seg_dict.get("words")
            )
            processed_segments.append(segment)
        return processed_segments

    def get_quality_recommendations(self, audio_duration: float) -> Dict[str, Any]:
        recommendations = {}
        recommended_quality_enum: TranscriptionQuality
        if audio_duration < 300: recommended_quality_enum = TranscriptionQuality.PREMIUM
        elif audio_duration < 1800: recommended_quality_enum = TranscriptionQuality.ACCURATE
        else: recommended_quality_enum = TranscriptionQuality.BALANCED
        recommendations["reason"] = f"Based on audio duration ({audio_duration/60:.1f} min)"

        if self.hardware_caps.device == "cpu":
            if recommended_quality_enum in [TranscriptionQuality.PREMIUM, TranscriptionQuality.ACCURATE]:
                recommended_quality_enum = TranscriptionQuality.BALANCED
                recommendations["reason"] += " (adjusted to BALANCED for CPU)"
            elif recommended_quality_enum == TranscriptionQuality.BALANCED:
                 recommended_quality_enum = TranscriptionQuality.FAST
                 recommendations["reason"] += " (adjusted to FAST for CPU)"
        recommendations["recommended_quality_value"] = recommended_quality_enum.value
        recommendations["recommended_config"] = self.default_configs.get(recommended_quality_enum)
        recommendations["all_options"] = {
            quality.value: {
                "estimated_time_minutes": round(self._estimate_processing_time(audio_duration, quality), 1),
                "description": self._get_quality_description(quality),
                "config_details": self.default_configs.get(quality).__dict__ if self.default_configs.get(quality) else None
            } for quality in TranscriptionQuality
        }
        return recommendations

    def _estimate_processing_time(self, audio_duration: float, quality: TranscriptionQuality) -> float:
        base_multiplier = {
            TranscriptionQuality.FAST: 0.05, TranscriptionQuality.BALANCED: 0.15,
            TranscriptionQuality.ACCURATE: 0.3, TranscriptionQuality.PREMIUM: 0.5
        }.get(quality, 0.2)
        hw_multiplier = 0.3 if self.hardware_caps.device == "cuda" and "H100" in (self.hardware_caps.gpu_name or "").upper() \
            else 0.6 if self.hardware_caps.device == "cuda" and (self.hardware_caps.gpu_memory_gb or 0) >= 20 \
            else 1.0 if self.hardware_caps.device == "cuda" and (self.hardware_caps.gpu_memory_gb or 0) >= 10 \
            else 1.5 if self.hardware_caps.device == "cuda" \
            else 4.0
        estimated_seconds = audio_duration * base_multiplier * hw_multiplier
        return max(0.1 * 60, estimated_seconds) / 60

    def _get_quality_description(self, quality: TranscriptionQuality) -> str:
        return {
            TranscriptionQuality.FAST: "Schnellste Transkription mit Basis-Genauigkeit...",
            TranscriptionQuality.BALANCED: "Gute Balance zwischen Geschwindigkeit und Genauigkeit...",
            TranscriptionQuality.ACCURATE: "Hohe Genauigkeit mit detaillierterer Verarbeitung...",
            TranscriptionQuality.PREMIUM: "Bestmögliche Qualität mit allen Optimierungen..."
        }.get(quality, "Standard Transkriptionsqualität.")

async def transcribe_audio_fast(audio: np.ndarray, sr: int, ts: TranscriptionService, mn: str = "large-v2") -> TranscriptionResult:
    return await ts.transcribe_audio(audio, sr, mn, TranscriptionConfig.get_fast_config())
async def transcribe_audio_balanced(audio: np.ndarray, sr: int, ts: TranscriptionService, mn: str = "large-v2") -> TranscriptionResult:
    return await ts.transcribe_audio(audio, sr, mn, TranscriptionConfig.get_balanced_config())
async def transcribe_audio_premium(audio: np.ndarray, sr: int, ts: TranscriptionService, hc: HardwareCapabilities, mn: str = "large-v2") -> TranscriptionResult:
    return await ts.transcribe_audio(audio, sr, mn, TranscriptionConfig.get_premium_config(hc))