# services/ml_pipeline.py
"""
ML Pipeline Orchestrator

Koordiniert alle ML-Services zu einer einheitlichen Pipeline.
Wie ein Dirigent, der das ganze Orchester aus Transkription,
Alignment, Diarization und Protokoll-Formatierung koordiniert.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

from services.transcription_service import (
    TranscriptionService, TranscriptionConfig, TranscriptionQuality, TranscriptionResult
)
from services.alignment_service import (
    AlignmentService, AlignmentConfig, AlignmentQuality, AlignmentResult
)
from services.diarization_service import (
    DiarizationService, DiarizationConfig, DiarizationQuality, DiarizationResult
)
from services.protocol_service import (
    ProtocolService, ProtocolConfig, ProtocolFormat, ProtocolResult
)
from config.hardware_detection import HardwareCapabilities
from config.model_config import ModelLoader

from services.enhanced_diarization_service import create_enhanced_diarization_service

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """Available pipeline processing modes"""
    FAST = "fast"               # Quick processing, basic quality
    BALANCED = "balanced"           # Good balance of quality and speed
    ACCURATE = "accurate"           # High quality, slower processing
    PREMIUM = "premium"             # Best quality, slowest processing
    CUSTOM = "custom"               # Custom configuration


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""

    # Overall settings
    mode: PipelineMode = PipelineMode.BALANCED

    # Individual service configs
    transcription_config: Optional[TranscriptionConfig] = None
    alignment_config: Optional[AlignmentConfig] = None
    diarization_config: Optional[DiarizationConfig] = None
    protocol_config: Optional[ProtocolConfig] = None

    # Processing options
    enable_alignment: bool = True
    enable_diarization: bool = True
    enable_protocol_formatting: bool = True

    # Model selection
    transcription_model: str = "large-v2"

    @classmethod
    def get_fast_config(cls) -> 'PipelineConfig':
        """Fast processing configuration"""
        return cls(
            mode=PipelineMode.FAST,
            transcription_config=TranscriptionConfig.get_fast_config(),
            alignment_config=AlignmentConfig.get_basic_config(),
            diarization_config=DiarizationConfig.get_fast_config(),
            protocol_config=ProtocolConfig.get_meeting_config(), # Sensible default
            enable_alignment=False,  # Skip for speed
            enable_diarization=True,
            transcription_model="base"  # Smaller, faster model
        )

    @classmethod
    def get_balanced_config(cls) -> 'PipelineConfig':
        """Balanced processing configuration"""
        return cls(
            mode=PipelineMode.BALANCED,
            transcription_config=TranscriptionConfig.get_balanced_config(),
            alignment_config=AlignmentConfig.get_precise_config(),
            diarization_config=DiarizationConfig.get_balanced_config(),
            protocol_config=ProtocolConfig.get_municipal_config(), # Default for balanced
            enable_alignment=True,
            enable_diarization=True,
            transcription_model="large-v2"
        )

    @classmethod
    def get_accurate_config(cls, hardware_caps: HardwareCapabilities) -> 'PipelineConfig':
        """High accuracy processing configuration"""
        return cls(
            mode=PipelineMode.ACCURATE,
            transcription_config=TranscriptionConfig.get_accurate_config(),
            alignment_config=AlignmentConfig.get_precise_config(),
            diarization_config=DiarizationConfig.get_accurate_config(),
            protocol_config=ProtocolConfig.get_municipal_config(),
            enable_alignment=True,
            enable_diarization=True,
            transcription_model="large-v2"
        )

    @classmethod
    def get_premium_config(cls, hardware_caps: HardwareCapabilities) -> 'PipelineConfig':
        """Premium quality processing configuration"""
        return cls(
            mode=PipelineMode.PREMIUM,
            transcription_config=TranscriptionConfig.get_premium_config(hardware_caps),
            alignment_config=AlignmentConfig.get_precise_config(), # Or PHONEME if available and desired
            diarization_config=DiarizationConfig.get_accurate_config(),
            protocol_config=ProtocolConfig.get_municipal_config(),
            enable_alignment=True,
            enable_diarization=True,
            transcription_model="large-v2" # Or a specific premium model if available
        )


@dataclass
class PipelineProgress:
    """Pipeline progress information"""
    stage: str
    progress: float  # 0.0 to 1.0
    message: str
    stage_start_time: float
    total_start_time: float

    def elapsed_time(self) -> float:
        """Total elapsed time"""
        return time.time() - self.total_start_time

    def stage_time(self) -> float:
        """Current stage elapsed time"""
        return time.time() - self.stage_start_time


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""

    # Fields without default values must come first
    transcription_result: TranscriptionResult
    config: PipelineConfig
    total_processing_time: float
    stage_times: Dict[str, float]
    original_audio_duration: float
    municipality: str
    meeting_date: str
    meeting_type: str

    # Fields with default values come after
    alignment_result: Optional[AlignmentResult] = None
    diarization_result: Optional[DiarizationResult] = None
    protocol_result: Optional[ProtocolResult] = None


    def get_final_result(self) -> Any:
        """Get the most processed result available"""
        if self.protocol_result:
            return self.protocol_result
        elif self.diarization_result:
            return self.diarization_result
        elif self.alignment_result:
            return self.alignment_result
        else:
            return self.transcription_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = {
            "pipeline_mode": self.config.mode.value,
            "total_processing_time": self.total_processing_time,
            "original_audio_duration": self.original_audio_duration,
            "processing_speed_ratio": self.original_audio_duration / self.total_processing_time if self.total_processing_time > 0 else 0,
            "stages_completed": list(self.stage_times.keys()),
            "stage_times": self.stage_times
        }

        # Add results from each stage
        if self.transcription_result:
            stats["transcription"] = self.transcription_result.get_statistics()

        if self.alignment_result:
            stats["alignment"] = self.alignment_result.get_statistics()

        if self.diarization_result:
            stats["diarization"] = self.diarization_result.get_statistics()

        if self.protocol_result:
            stats["protocol"] = self.protocol_result.get_statistics()

        return stats


class MLPipelineOrchestrator:
    """
    Main ML Pipeline Orchestrator.

    Der Generaldirigent des gesamten ML-Orchesters - koordiniert alle Services
    und sorgt dafür, dass sie harmonisch zusammenspielen.
    """

    def __init__(
        self,
        hardware_caps: HardwareCapabilities,
        model_loader: ModelLoader,
        model_cache_dir: str, # model_cache_dir for AlignmentService
        hf_token: Optional[str] = None
    ):
        self.hardware_caps = hardware_caps

        # Initialize all services
        self.transcription_service = TranscriptionService(model_loader, hardware_caps)
        self.alignment_service = AlignmentService(hardware_caps, model_cache_dir) # Pass model_cache_dir
        self.protocol_service = ProtocolService()
        self.diarization_service = create_enhanced_diarization_service(hardware_caps, hf_token)

        # Pipeline state
        self.current_progress: Optional[PipelineProgress] = None
        self.is_processing = False
        self._pipeline_start_time: float = 0.0 # To store the overall pipeline start time

    def _notify_progress(
        self,
        stage: str,
        progress: float, # Stage-specific progress (0.0 to 1.0 for that stage)
        message: str,
        stage_internal_start_time: float, # When this specific stage started (for stage_time)
        callback: Optional[Callable[[PipelineProgress], None]],
        overall_progress_weight: float, # How much this stage contributes to overall (e.g., 0.4 for transcription)
        overall_progress_offset: float  # Offset for this stage in overall progress (e.g., 0.0 for transp, 0.4 for align)
    ):
        """Notify progress to callback, calculating overall progress."""
        if callback:
            # Calculate progress relative to the entire pipeline
            current_overall_progress = overall_progress_offset + (progress * overall_progress_weight)
            current_overall_progress = min(max(current_overall_progress, 0.0), 1.0) # Clamp between 0 and 1

            progress_info = PipelineProgress(
                stage=stage,
                progress=current_overall_progress, # Report overall progress
                message=message,
                stage_start_time=stage_internal_start_time, # For calculating current stage duration
                total_start_time=self._pipeline_start_time # For calculating total elapsed time
            )

            self.current_progress = progress_info

            try:
                callback(progress_info)
            except Exception as e:
                logger.warning(f"⚠️ Progress callback error: {e}")

    async def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        municipality: str,
        meeting_date: str,
        meeting_type: str,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None
    ) -> PipelineResult:
        """
        Process audio through the complete ML pipeline.
        """
        effective_config = config or PipelineConfig.get_balanced_config()
        self._pipeline_start_time = time.time() # Set overall pipeline start time
        stage_times = {}

        logger.info(
            f"🚀 Starting ML pipeline: {effective_config.mode.value} mode, "
            f"audio={len(audio)/sample_rate:.1f}s"
        )
        self.is_processing = True

        # Define progress weights for each stage (must sum to ~1.0 if all enabled)
        # These are approximate contributions to the total time/effort.
        # Example: Transcription 40%, Alignment 20%, Diarization 20%, Protocol 15%, Finalizing 5%
        progress_weights = {
            "transcription": 0.40,
            "alignment": 0.20,
            "diarization": 0.20,
            "protocol_formatting": 0.15,
            "finalizing": 0.05
        }
        current_progress_offset = 0.0

        try:
            # === STAGE 1: TRANSCRIPTION ===
            stage_internal_start_time = time.time()
            self._notify_progress("transcription_init", 0.0, "Starting transcription", stage_internal_start_time, progress_callback, progress_weights["transcription"], current_progress_offset)

            def transcription_stage_progress(stage_name_detail: str, stage_specific_progress: float):
                # stage_specific_progress is 0.0-1.0 for the transcription service itself
                self._notify_progress(
                    f"transcription.{stage_name_detail}", stage_specific_progress,
                    f"Transcription: {stage_name_detail}", stage_internal_start_time, progress_callback,
                    progress_weights["transcription"], current_progress_offset
                )

            transcription_result = await self.transcription_service.transcribe_audio(
                audio=audio,
                sample_rate=sample_rate,
                model_name=effective_config.transcription_model,
                config=effective_config.transcription_config,
                progress_callback=transcription_stage_progress
            )
            stage_times["transcription"] = time.time() - stage_internal_start_time
            logger.info(f"📝 Transcription: {transcription_result.segments_count} segments, {transcription_result.words_count} words in {stage_times['transcription']:.2f}s")
            current_progress_offset += progress_weights["transcription"]


            # === STAGE 2: ALIGNMENT (Optional) ===
            alignment_result = None
            if effective_config.enable_alignment:
                stage_internal_start_time = time.time()
                self._notify_progress("alignment_init", 0.0, "Starting alignment", stage_internal_start_time, progress_callback, progress_weights["alignment"], current_progress_offset)

                def alignment_stage_progress(stage_name_detail: str, stage_specific_progress: float):
                    self._notify_progress(
                        f"alignment.{stage_name_detail}", stage_specific_progress,
                        f"Alignment: {stage_name_detail}", stage_internal_start_time, progress_callback,
                        progress_weights["alignment"], current_progress_offset
                    )

                alignment_result = await self.alignment_service.align_transcription_result(
                    transcription_result=transcription_result,
                    audio=audio,
                    sample_rate=sample_rate,
                    config=effective_config.alignment_config,
                    progress_callback=alignment_stage_progress
                )
                stage_times["alignment"] = time.time() - stage_internal_start_time
                logger.info(f"🎯 Alignment: {alignment_result.total_words} words aligned in {stage_times['alignment']:.2f}s")
                current_progress_offset += progress_weights["alignment"]
            else:
                logger.info("➡️ Alignment skipped by configuration.")


            # === STAGE 3: DIARIZATION (Optional) ===
            diarization_result = None
            if effective_config.enable_diarization:
                if not self.diarization_service.is_available:
                    logger.warning("⚠️ Diarization service not available (missing HF token?) - skipping.")
                else:
                    stage_internal_start_time = time.time()
                    self._notify_progress("diarization_init", 0.0, "Starting speaker diarization", stage_internal_start_time, progress_callback, progress_weights["diarization"], current_progress_offset)

                    def diarization_stage_progress(stage_name_detail: str, stage_specific_progress: float):
                        self._notify_progress(
                            f"diarization.{stage_name_detail}", stage_specific_progress,
                            f"Diarization: {stage_name_detail}", stage_internal_start_time, progress_callback,
                            progress_weights["diarization"], current_progress_offset
                        )

                    if alignment_result: # Prefer aligned result for diarization if available
                        diarization_result = await self.diarization_service.diarize_alignment_result(
                            alignment_result=alignment_result,
                            audio=audio,
                            sample_rate=sample_rate,
                            config=effective_config.diarization_config,
                            progress_callback=diarization_stage_progress
                        )
                    else:
                        diarization_result = await self.diarization_service.diarize_transcription_result(
                            transcription_result=transcription_result,
                            audio=audio,
                            sample_rate=sample_rate,
                            config=effective_config.diarization_config,
                            progress_callback=diarization_stage_progress
                        )
                    stage_times["diarization"] = time.time() - stage_internal_start_time
                    if diarization_result:
                        logger.info(f"👥 Diarization: {diarization_result.speaker_count} speakers, {len(diarization_result.segments)} segments in {stage_times['diarization']:.2f}s")
                    current_progress_offset += progress_weights["diarization"]
            else:
                logger.info("➡️ Diarization skipped by configuration.")

            # === STAGE 4: PROTOCOL FORMATTING (Optional) ===
            protocol_result = None
            if effective_config.enable_protocol_formatting:
                stage_internal_start_time = time.time()
                self._notify_progress("protocol_formatting_init", 0.0, "Formatting protocol", stage_internal_start_time, progress_callback, progress_weights["protocol_formatting"], current_progress_offset)

                # Prepare metadata for protocol service
                protocol_metadata = {
                    "transcription_model": effective_config.transcription_model,
                    "pipeline_mode": effective_config.mode.value,
                    "processing_timestamp": time.time(), # Current time
                    "hardware_device": self.hardware_caps.device,
                    "original_audio_duration_seconds": len(audio) / sample_rate
                }
                if self.hardware_caps.gpu_name:
                    protocol_metadata["gpu_name"] = self.hardware_caps.gpu_name


                def format_protocol_sync(): # Wrapper for asyncio.to_thread
                    if diarization_result:
                        return self.protocol_service.format_diarized_transcription(
                            diarization_result=diarization_result,
                            municipality=municipality,
                            meeting_date=meeting_date,
                            meeting_type=meeting_type,
                            config=effective_config.protocol_config,
                            metadata=protocol_metadata
                        )
                    else:
                        return self.protocol_service.format_simple_transcription(
                            transcription_result=transcription_result,
                            municipality=municipality,
                            meeting_date=meeting_date,
                            meeting_type=meeting_type,
                            config=effective_config.protocol_config
                            # metadata can be passed here too if needed for simple transcription
                        )

                protocol_result = await asyncio.to_thread(format_protocol_sync)
                stage_times["protocol_formatting"] = time.time() - stage_internal_start_time
                if protocol_result:
                     logger.info(
                        f"📄 Protocol: {protocol_result.total_words} words, "
                        f"{len(protocol_result.topics)} topics, {len(protocol_result.decisions)} decisions in {stage_times['protocol_formatting']:.2f}s"
                    )
                current_progress_offset += progress_weights["protocol_formatting"]
            else:
                logger.info("➡️ Protocol formatting skipped by configuration.")


            # === FINALIZING ===
            self._notify_progress("finalizing", 1.0, "Pipeline completed successfully", time.time(), progress_callback, progress_weights["finalizing"], current_progress_offset)

            total_pipeline_time = time.time() - self._pipeline_start_time

            final_result = PipelineResult(
                transcription_result=transcription_result,
                alignment_result=alignment_result,
                diarization_result=diarization_result,
                protocol_result=protocol_result,
                config=effective_config,
                total_processing_time=total_pipeline_time,
                stage_times=stage_times,
                original_audio_duration=len(audio) / sample_rate,
                municipality=municipality,
                meeting_date=meeting_date,
                meeting_type=meeting_type
            )

            logger.info(
                f"✅ Pipeline completed: {total_pipeline_time:.2f}s total, "
                f"speed ratio: {final_result.get_statistics()['processing_speed_ratio']:.1f}x"
            )
            return final_result

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            # Notify failure if callback is provided
            if progress_callback:
                 self._notify_progress("error", self.current_progress.progress if self.current_progress else 0.0 , f"Pipeline failed: {e}", time.time(), progress_callback, 0, self.current_progress.progress if self.current_progress else 0.0) # Ensure progress doesn't exceed 1.0
            raise # Re-raise the exception to be caught by the caller

        finally:
            self.is_processing = False


    def get_pipeline_recommendations(
        self,
        audio_duration: float,
        expected_speakers: Optional[int] = None,
        meeting_type: str = "Stadtrat",
        quality_priority: str = "balanced"  # "speed", "balanced", "quality"
    ) -> Dict[str, Any]:
        """
        Get pipeline configuration recommendations.
        """
        recommendations = {
            "hardware_info": {
                "device": self.hardware_caps.device,
                "gpu_name": self.hardware_caps.gpu_name,
                "gpu_memory_gb": self.hardware_caps.gpu_memory_gb
            },
            "service_availability": {
                "transcription": True,
                "alignment": True, # Assuming alignment service is always available
                "diarization": self.diarization_service.is_available,
                "protocol_formatting": True # Assuming protocol service is always available
            }
        }

        # Recommend pipeline mode based on priority and hardware
        if quality_priority == "speed":
            if audio_duration > 1800:  # > 30 minutes
                recommended_mode = PipelineMode.FAST
            else:
                recommended_mode = PipelineMode.BALANCED
        elif quality_priority == "quality":
            if self.hardware_caps.device == "cuda" and audio_duration < 7200: # Allow longer for quality if GPU
                recommended_mode = PipelineMode.PREMIUM
            else:
                recommended_mode = PipelineMode.ACCURATE
        else:  # balanced
            recommended_mode = PipelineMode.BALANCED

        recommendations["recommended_pipeline_mode"] = recommended_mode.value # Return the string value

        # Get service-specific recommendations
        # Ensure the services are initialized before calling their recommendation methods
        if hasattr(self.transcription_service, 'get_quality_recommendations'):
             recommendations["transcription"] = self.transcription_service.get_quality_recommendations(audio_duration)

        if hasattr(self.alignment_service, 'get_alignment_recommendations'):
            recommendations["alignment"] = self.alignment_service.get_alignment_recommendations(
                audio_duration, expected_speakers or 5, "de" # Assuming default language "de"
            )

        if self.diarization_service.is_available and hasattr(self.diarization_service, 'get_diarization_recommendations'):
            recommendations["diarization"] = self.diarization_service.get_diarization_recommendations(
                audio_duration, expected_speakers
            )

        if hasattr(self.protocol_service, 'get_format_recommendations'):
            recommendations["protocol"] = self.protocol_service.get_format_recommendations(
                meeting_type, audio_duration / 60, bool(expected_speakers)
            )

        # Estimate total processing time
        base_times = {
            PipelineMode.FAST: 0.3,
            PipelineMode.BALANCED: 0.5,
            PipelineMode.ACCURATE: 0.8,
            PipelineMode.PREMIUM: 1.2
        }

        hardware_multiplier = 0.6 if self.hardware_caps.device == "cuda" else 2.0
        # Use the determined recommended_mode for estimation
        estimated_minutes = (audio_duration / 60) * base_times.get(recommended_mode, 0.5) * hardware_multiplier

        recommendations["estimated_processing_time_minutes"] = round(max(0.5, estimated_minutes), 1)

        return recommendations

    def cleanup_resources(self):
        """Clean up all service resources"""
        logger.info("🧹 Cleaning up ML pipeline resources...")

        try:
            if hasattr(self.alignment_service, 'clear_internal_cache'):
                self.alignment_service.clear_internal_cache()
            if hasattr(self.diarization_service, 'cleanup_resources'):
                self.diarization_service.cleanup_resources()
            # Transcription models are typically managed by ModelLoader or cleaned up by TranscriptionService itself
            if hasattr(self.transcription_service, 'cleanup_model'): # Assuming such a method might exist
                 self.transcription_service.cleanup_model()
            logger.info("✅ ML pipeline cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


# Convenience functions for common use cases
async def process_audio_fast(
    audio: np.ndarray,
    sample_rate: int,
    municipality: str,
    meeting_date: str,
    meeting_type: str,
    pipeline: MLPipelineOrchestrator, # Pass the orchestrator instance
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """Fast audio processing"""
    config = PipelineConfig.get_fast_config()
    return await pipeline.process_audio(
        audio, sample_rate, municipality, meeting_date, meeting_type, config, progress_callback
    )


async def process_audio_balanced(
    audio: np.ndarray,
    sample_rate: int,
    municipality: str,
    meeting_date: str,
    meeting_type: str,
    pipeline: MLPipelineOrchestrator,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """Balanced audio processing"""
    config = PipelineConfig.get_balanced_config()
    return await pipeline.process_audio(
        audio, sample_rate, municipality, meeting_date, meeting_type, config, progress_callback
    )


async def process_audio_premium(
    audio: np.ndarray,
    sample_rate: int,
    municipality: str,
    meeting_date: str,
    meeting_type: str,
    pipeline: MLPipelineOrchestrator,
    hardware_caps: HardwareCapabilities, # Needed for premium config
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """Premium quality audio processing"""
    config = PipelineConfig.get_premium_config(hardware_caps)
    return await pipeline.process_audio(
        audio, sample_rate, municipality, meeting_date, meeting_type, config, progress_callback
    )


async def process_municipal_meeting(
    audio: np.ndarray,
    sample_rate: int,
    municipality: str,
    meeting_date: str,
    pipeline: MLPipelineOrchestrator,
    expected_speakers: int = 6,
    progress_callback: Optional[Callable[[PipelineProgress], None]] = None
) -> PipelineResult:
    """Process municipal council meeting with optimized settings"""

    config = PipelineConfig.get_balanced_config() # Start with balanced

    # Optimize diarization for expected speakers
    if config.diarization_config:
        # Use the specific meeting config for diarization
        meeting_diar_config = DiarizationConfig.get_meeting_config(expected_speakers)
        config.diarization_config = meeting_diar_config
    else: # If no diarization_config was set by get_balanced_config, create one
        config.diarization_config = DiarizationConfig.get_meeting_config(expected_speakers)


    # Use municipal protocol format
    if config.protocol_config:
        config.protocol_config.format_type = ProtocolFormat.MUNICIPAL_COUNCIL
    else:
        config.protocol_config = ProtocolConfig.get_municipal_config()


    return await pipeline.process_audio(
        audio, sample_rate, municipality, meeting_date, "Stadtrat", config, progress_callback
    )