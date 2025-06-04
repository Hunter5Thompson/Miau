# services/diarization_service.py
"""
Speaker Diarization Service

Identifiziert verschiedene Sprecher in Audio-Aufnahmen und ordnet
Transkriptionssegmente den entsprechenden Sprechern zu.
Wie ein erfahrener Moderator, der auch bei lebhaften Diskussionen
immer weiß, wer gerade das Wort hat.
"""

import asyncio
import logging
import time
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os

import numpy as np
import torch
import whisperx
from services.transcription_service import TranscriptionResult
from services.alignment_service import AlignmentResult, AlignedSegment
from config.hardware_detection import HardwareCapabilities

logger = logging.getLogger(__name__)


class DiarizationQuality(Enum):
    """Diarization quality levels"""
    FAST = "fast"          # Quick speaker detection
    BALANCED = "balanced"  # Good balance of accuracy and speed
    ACCURATE = "accurate"  # High accuracy speaker identification


class SpeakerLabelMode(Enum):
    """Speaker labeling modes"""
    SPEAKER_ID = "speaker_id"    # SPEAKER_00, SPEAKER_01, etc.
    DESCRIPTIVE = "descriptive"  # Speaker A, Speaker B, etc.
    NUMBERED = "numbered"        # Speaker 1, Speaker 2, etc.
    CUSTOM = "custom"            # Custom naming scheme


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization"""

    # Quality settings
    quality: DiarizationQuality = DiarizationQuality.BALANCED

    # Speaker detection parameters
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Clustering parameters
    clustering_method: str = "spectral"  # spectral, agglomerative
    min_segment_length: float = 1.0      # Minimum segment length in seconds

    # Speaker labeling
    speaker_label_mode: SpeakerLabelMode = SpeakerLabelMode.SPEAKER_ID
    custom_speaker_names: Optional[List[str]] = None

    # Advanced settings
    enhance_diarization: bool = True     # Use voice activity detection enhancement
    merge_threshold: float = 0.5         # Threshold for merging adjacent segments

    @classmethod
    def get_fast_config(cls, max_speakers: int = 6) -> 'DiarizationConfig':
        """Fast diarization configuration"""
        return cls(
            quality=DiarizationQuality.FAST,
            max_speakers=max_speakers,
            clustering_method="agglomerative",
            min_segment_length=0.5,
            enhance_diarization=False
        )

    @classmethod
    def get_balanced_config(cls, max_speakers: int = 8) -> 'DiarizationConfig':
        """Balanced diarization configuration"""
        return cls(
            quality=DiarizationQuality.BALANCED,
            max_speakers=max_speakers,
            clustering_method="spectral",
            min_segment_length=1.0,
            enhance_diarization=True
        )

    @classmethod
    def get_accurate_config(cls, max_speakers: int = 10) -> 'DiarizationConfig':
        """High accuracy diarization configuration"""
        return cls(
            quality=DiarizationQuality.ACCURATE,
            max_speakers=max_speakers,
            clustering_method="spectral",
            min_segment_length=1.5,
            enhance_diarization=True,
            merge_threshold=0.3
        )

    @classmethod
    def get_meeting_config(cls, expected_speakers: int = 6) -> 'DiarizationConfig':
        """Configuration optimized for meeting recordings"""
        return cls(
            quality=DiarizationQuality.ACCURATE,
            min_speakers=2,
            max_speakers=min(expected_speakers + 2, 12),  # Allow some flexibility
            clustering_method="spectral",
            min_segment_length=2.0,  # Longer segments for formal meetings
            speaker_label_mode=SpeakerLabelMode.DESCRIPTIVE,
            enhance_diarization=True,
            merge_threshold=0.4
        )


@dataclass
class SpeakerSegment:
    """Segment assigned to a specific speaker"""
    speaker_id: str
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[Dict[str, Any]]] = None

    def duration(self) -> float:
        """Duration of the speaker segment"""
        return self.end - self.start

    def word_count(self) -> int:
        """Number of words in this segment"""
        return len(self.text.split()) if self.text else 0


@dataclass
class SpeakerProfile:
    """Profile information for a detected speaker"""
    speaker_id: str
    speaker_label: str
    total_speaking_time: float
    segment_count: int
    average_segment_duration: float
    confidence: Optional[float] = None

    def speaking_percentage(self, total_duration: float) -> float:
        """Percentage of total audio this speaker was speaking"""
        if total_duration > 0:
            return (self.total_speaking_time / total_duration) * 100
        return 0.0


@dataclass
class DiarizationResult:
    """Complete diarization result"""
    segments: List[SpeakerSegment]
    speakers: List[SpeakerProfile]
    total_duration: float
    config: DiarizationConfig

    # Processing metadata
    processing_time: float
    model_load_time: Optional[float] = None
    diarization_time: Optional[float] = None
    assignment_time: Optional[float] = None

    # Quality metrics
    speaker_count: int = 0
    coverage_ratio: float = 1.0  # Ratio of audio assigned to speakers

    def __post_init__(self):
        """Calculate derived metrics"""
        self.speaker_count = len(self.speakers)

        # Calculate coverage ratio
        assigned_duration = sum(seg.duration() for seg in self.segments)
        if self.total_duration > 0:
            self.coverage_ratio = assigned_duration / self.total_duration

    def get_speaker_by_id(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get speaker profile by ID"""
        for speaker in self.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None

    def get_segments_by_speaker(self, speaker_id: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker"""
        return [seg for seg in self.segments if seg.speaker_id == speaker_id]

    def get_speaker_timeline(self) -> List[Tuple[float, float, str]]:
        """Get chronological timeline of speaker changes"""
        timeline = [(seg.start, seg.end, seg.speaker_id) for seg in self.segments]
        return sorted(timeline, key=lambda x: x[0])

    def get_statistics(self) -> Dict[str, Any]:
        """Get diarization statistics"""
        stats = {
            "total_duration_seconds": self.total_duration,
            "processing_time_seconds": self.processing_time,
            "speaker_count": self.speaker_count,
            "segments_count": len(self.segments),
            "coverage_ratio": self.coverage_ratio,
            "quality_level": self.config.quality.value
        }

        # Add per-speaker statistics
        stats["speakers"] = []
        for speaker in self.speakers:
            speaker_stats = {
                "speaker_id": speaker.speaker_id,
                "speaker_label": speaker.speaker_label,
                "speaking_time_seconds": speaker.total_speaking_time,
                "speaking_percentage": speaker.speaking_percentage(self.total_duration),
                "segment_count": speaker.segment_count,
                "average_segment_duration": speaker.average_segment_duration
            }
            stats["speakers"].append(speaker_stats)

        return stats


class SpeakerLabeler:
    """Handles speaker labeling and naming"""

    @staticmethod
    def generate_labels(
        speaker_ids: List[str], 
        mode: SpeakerLabelMode,
        custom_names: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate speaker labels based on mode.

        Args:
            speaker_ids: List of speaker IDs from diarization
            mode: Labeling mode
            custom_names: Custom names (for CUSTOM mode)

        Returns:
            Dict mapping speaker_id to label
        """
        labels = {}

        if mode == SpeakerLabelMode.SPEAKER_ID:
            # Keep original IDs (SPEAKER_00, SPEAKER_01, etc.)
            for speaker_id in speaker_ids:
                labels[speaker_id] = speaker_id

        elif mode == SpeakerLabelMode.DESCRIPTIVE:
            # Speaker A, Speaker B, etc.
            for i, speaker_id in enumerate(sorted(speaker_ids)):
                letter = chr(ord('A') + i)
                labels[speaker_id] = f"Speaker {letter}"

        elif mode == SpeakerLabelMode.NUMBERED:
            # Speaker 1, Speaker 2, etc.
            for i, speaker_id in enumerate(sorted(speaker_ids)):
                labels[speaker_id] = f"Speaker {i + 1}"

        elif mode == SpeakerLabelMode.CUSTOM:
            # Custom names
            if custom_names:
                for i, speaker_id in enumerate(sorted(speaker_ids)):
                    if i < len(custom_names):
                        labels[speaker_id] = custom_names[i]
                    else:
                        # Fallback for extra speakers
                        labels[speaker_id] = f"Speaker {i + 1}"
            else:
                # Fallback to numbered if no custom names provided
                return SpeakerLabeler.generate_labels(
                    speaker_ids, SpeakerLabelMode.NUMBERED
                )

        return labels


class DiarizationEngine:
    """Core diarization engine using PyAnnote/WhisperX"""

    def __init__(self, hardware_caps: HardwareCapabilities, hf_token: Optional[str] = None):
        self.hardware_caps = hardware_caps
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.diarization_pipeline = None
        self.pipeline_load_time = 0.0

        if not self.hf_token:
            logger.warning("⚠️ No HuggingFace token provided. Diarization may not work.")

    async def _ensure_pipeline_loaded(self, config: DiarizationConfig):
        """Ensure diarization pipeline is loaded"""
        if self.diarization_pipeline is None:
            logger.info("📦 Loading diarization pipeline...")

            def _load_pipeline():
                return whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.hardware_caps.device
                )

            start_time = time.time()
            self.diarization_pipeline = await asyncio.to_thread(_load_pipeline)
            self.pipeline_load_time = time.time() - start_time

            logger.info(f"✅ Diarization pipeline loaded in {self.pipeline_load_time:.2f}s")
            # Optional: callback here if diarization_pipeline itself has progress during init
            # if progress_callback:
            # progress_callback("model_loaded", 1.0) # completion of this sub-step

    async def diarize_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: DiarizationConfig,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data as a NumPy array
            sample_rate: Sample rate of the audio
            config: Diarization configuration
            progress_callback: Progress callback

        Returns:
            Dict: Raw diarization results including segments and pipeline_load_time
        """
        if not self.hf_token:
            raise ValueError("HuggingFace token required for speaker diarization")

        logger.info(f"👥 Starting diarization: {config.quality.value} quality")

        # === STEP 1: LOAD DIARIZATION PIPELINE ===
        if progress_callback:
            # Progress is a value from 0.0 to 1.0 for this specific sub-task (diarize_audio)
            # The outer call (DiarizationService) might scale this.
            progress_callback("loading_diarization_model", 0.1) 
        await self._ensure_pipeline_loaded(config)
        # self.pipeline_load_time is now set if loading occurred.

        # === STEP 2: PREPARE AUDIO INPUT ===
        if progress_callback:
            progress_callback("preparing_audio", 0.3)

        # Ensure audio is float32 tensor and has a batch dimension
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # The whisperx.DiarizationPipeline often takes the audio file path or a pre-loaded audio tensor.
        # PyAnnote pipelines (which whisperx.DiarizationPipeline wraps) typically expect a dict like this:
        diarize_input = {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
        # However, whisperx documentation often shows `pipeline(audio_file)` or `pipeline(waveform_tensor)`
        # Let's assume `whisperx.DiarizationPipeline` is flexible. If it expects just the tensor:
        # diarize_input_for_pipeline = audio_tensor
        # If it expects the dict:
        diarize_input_for_pipeline = diarize_input


        # === STEP 3: EXECUTE DIARIZATION ===
        if progress_callback:
            progress_callback("diarizing", 0.4)

        def _diarize_task():
            if self.diarization_pipeline is None:
                logger.error("Diarization pipeline not loaded before diarization call.")
                raise RuntimeError("Diarization pipeline failed to load.")
            
            # According to whisperx examples, it's usually `pipeline(audio_file_path)` or `pipeline(audio_tensor)`
            # If `audio` is already loaded (as np.array then torch.tensor), we pass the tensor.
            # The `whisperx.DiarizationPipeline` call signature might vary slightly from pure pyannote.
            # It seems it can take the audio file directly, or the loaded audio.
            # Based on common whisperX usage, we'd pass the audio directly.
            # If it were a raw pyannote pipeline, it would be `pipeline(diarize_input_for_pipeline)`
            # Let's stick to the `whisperx` way of calling its DiarizationPipeline,
            # which is typically `diarization_pipeline(audio_file_path_or_waveform, min_speakers=, max_speakers=)`
            # Here, `audio` is the raw numpy array. The pipeline might handle the tensor conversion internally,
            # or it might expect the tensor directly.
            # Given the structure of `whisperx` functions, passing the original audio `np.ndarray`
            # might be what it expects if it's designed like `whisperx.load_model` and then `model.transcribe(audio_np_array)`.
            # However, `DiarizationPipeline` comes from `pyannote.audio` which uses waveform tensors.
            # The most robust way is to provide what `pyannote.audio.Pipeline.__call__` expects:
            # a dictionary `{"waveform": tensor, "sample_rate": sr}` or just the waveform tensor if sample rate is known/fixed.
            # The previous broken snippet implied `self.diarization_pipeline(diarize_input_dict, ...)`
            # This is consistent with pyannote.audio.Pipeline usage.
            
            # Assuming `whisperx.DiarizationPipeline` follows `pyannote.audio.Pipeline`'s call signature for preloaded audio:
            return self.diarization_pipeline(
                {"waveform": audio_tensor, "sample_rate": sample_rate}, # Pass the dict
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers
            )

        diarize_segments_annotation = await asyncio.to_thread(_diarize_task)
        # diarize_segments_annotation is a pyannote.core.Annotation object

        # === STEP 4: PROCESS DIARIZATION RESULTS ===
        if progress_callback:
            progress_callback("processing_results", 0.8)

        processed_results = self._process_diarization_output(diarize_segments_annotation, config)

        logger.info(f"✅ Diarization completed: {len(processed_results.get('segments', []))} segments")

        return processed_results

    def _process_diarization_output(
        self, 
        diarize_segments_annotation: Any, # pyannote.core.Annotation object
        config: DiarizationConfig
    ) -> Dict[str, Any]:
        """Process raw diarization output from pyannote.core.Annotation"""
        segments = []

        # Extract segments from diarization result (pyannote.core.Annotation)
        for turn, _, speaker in diarize_segments_annotation.itertracks(yield_label=True):
            segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)  # speaker is the label like "SPEAKER_00"
            })

        # Post-process segments based on configuration
        if config.min_segment_length > 0:
            segments = self._filter_short_segments(segments, config.min_segment_length)

        if config.merge_threshold > 0:
            segments = self._merge_adjacent_segments(segments, config.merge_threshold)

        return {
            "segments": segments,
            "pipeline_load_time": self.pipeline_load_time # Pass this along
        }

    def _filter_short_segments(
        self, 
        segments: List[Dict[str, Any]], 
        min_length: float
    ) -> List[Dict[str, Any]]:
        """Filter out segments shorter than minimum length"""
        filtered = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration >= min_length:
                filtered.append(segment)
            else:
                logger.debug(f"Filtered short segment: {duration:.2f}s < {min_length:.2f}s")

        if len(segments) != len(filtered):
            logger.info(f"🔍 Filtered short segments: {len(segments)} -> {len(filtered)}")
        return filtered

    def _merge_adjacent_segments(
        self, 
        segments: List[Dict[str, Any]], 
        merge_threshold: float
    ) -> List[Dict[str, Any]]:
        """Merge adjacent segments from the same speaker"""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            # Check if segments are from same speaker and close together
            same_speaker = current_segment["speaker"] == next_segment["speaker"]
            gap = next_segment["start"] - current_segment["end"]
            close_together = gap <= merge_threshold

            if same_speaker and close_together:
                # Merge segments
                current_segment["end"] = next_segment["end"]
                logger.debug(f"Merged segments for {current_segment['speaker']} with gap: {gap:.2f}s")
            else:
                # Save current segment and start new one
                merged.append(current_segment)
                current_segment = next_segment.copy()

        # Don't forget the last segment
        merged.append(current_segment)

        if len(segments) != len(merged):
            logger.info(f"🔗 Merged adjacent segments: {len(segments)} -> {len(merged)}")
        return merged

    def cleanup_pipeline(self):
        """Clean up diarization pipeline"""
        if self.diarization_pipeline is not None:
            try:
                del self.diarization_pipeline
                self.diarization_pipeline = None
                gc.collect()
                if self.hardware_caps.device == "cuda":
                    torch.cuda.empty_cache()
                logger.info("🧹 Diarization pipeline cleaned up")
            except Exception as e:
                logger.warning(f"⚠️ Pipeline cleanup warning: {e}")


class SpeakerAssigner:
    """Assigns transcription segments to speakers"""

    @staticmethod
    def assign_speakers_to_transcription(
        transcription_segments: List[Dict[str, Any]],
        diarization_segments: List[Dict[str, Any]],
        config: DiarizationConfig # config might not be needed here unless specific assignment logic uses it
    ) -> List[Dict[str, Any]]:
        """
        Assign speakers to transcription segments based on diarization.

        Args:
            transcription_segments: Segments from transcription
            diarization_segments: Speaker segments from diarization
            config: Diarization configuration (currently unused in this method)

        Returns:
            List[Dict]: Transcription segments with speaker assignments
        """
        logger.info(
            f"🎯 Assigning speakers: {len(transcription_segments)} transcription segments, "
            f"{len(diarization_segments)} diarization segments"
        )

        assigned_segments = []

        for trans_seg in transcription_segments:
            trans_start = float(trans_seg.get("start", 0))
            trans_end = float(trans_seg.get("end", 0))
            # Ensure start is not after end, which could lead to negative duration
            if trans_start > trans_end: 
                logger.warning(f"Transcription segment with invalid times: start={trans_start}, end={trans_end}. Skipping.")
                # Create assigned segment even if problematic, marking as unknown
                assigned_seg = trans_seg.copy()
                assigned_seg["speaker"] = "SPEAKER_UNKNOWN"
                assigned_segments.append(assigned_seg)
                continue

            trans_mid = (trans_start + trans_end) / 2
            
            best_speaker = None
            max_overlap = -1.0 # Use -1 to ensure any valid overlap is chosen

            for diar_seg in diarization_segments:
                diar_start = float(diar_seg.get("start", 0))
                diar_end = float(diar_seg.get("end", 0))
                
                # Calculate overlap: intersection of [trans_start, trans_end] and [diar_start, diar_end]
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                if overlap_duration > 0:
                    # Prioritize segment with maximum overlap
                    if overlap_duration > max_overlap:
                        max_overlap = overlap_duration
                        best_speaker = diar_seg.get("speaker")
                elif max_overlap <= 0: # Only consider midpoint if no overlap found yet
                    # Fallback: if midpoint of transcription segment falls within a diarization segment
                    if diar_start <= trans_mid < diar_end: # Use < for end to handle segments of zero duration correctly
                        best_speaker = diar_seg.get("speaker")
                        # No actual overlap, so don't update max_overlap based on this rule alone
                        # This ensures that if a tiny overlap exists, it's preferred over midpoint rule

            assigned_seg = trans_seg.copy()
            if best_speaker:
                assigned_seg["speaker"] = best_speaker
            else:
                assigned_seg["speaker"] = "SPEAKER_UNKNOWN"
                logger.debug(f"No speaker found for segment {trans_start:.2f}-{trans_end:.2f}s. Midpoint: {trans_mid:.2f}s")
            
            assigned_segments.append(assigned_seg)

        speakers_found = len(set(seg.get("speaker") for seg in assigned_segments if seg.get("speaker") != "SPEAKER_UNKNOWN"))
        unknown_count = len([seg for seg in assigned_segments if seg.get("speaker") == "SPEAKER_UNKNOWN"])

        logger.info(
            f"✅ Speaker assignment complete: {speakers_found} distinct speakers identified, "
            f"{unknown_count} segments unassigned or marked SPEAKER_UNKNOWN"
        )

        return assigned_segments


class DiarizationService:
    """
    Main speaker diarization service.

    Der Gesprächs-Analytiker - wie ein erfahrener Journalist, der auch bei
    hitzigen Diskussionen immer den Überblick behält und jeden Sprecher
    klar identifizieren kann.
    """

    def __init__(self, hardware_caps: HardwareCapabilities, hf_token: Optional[str] = None):
        self.engine = DiarizationEngine(hardware_caps, hf_token)
        self.assigner = SpeakerAssigner()
        self.labeler = SpeakerLabeler()
        self.hardware_caps = hardware_caps

        self.is_available = bool(hf_token or os.getenv("HUGGINGFACE_TOKEN"))
        if not self.is_available:
            logger.warning("⚠️ Diarization service not available - missing HuggingFace token")

    async def diarize_transcription_result(
        self,
        transcription_result: TranscriptionResult,
        audio: np.ndarray,
        sample_rate: int,
        config: Optional[DiarizationConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> DiarizationResult:
        """
        Add speaker diarization to transcription result.

        Args:
            transcription_result: Result from transcription service
            audio: Original audio data
            sample_rate: Audio sample rate
            config: Diarization configuration
            progress_callback: Progress callback

        Returns:
            DiarizationResult: Transcription with speaker assignments
        """
        if not self.is_available:
            raise ValueError("Diarization service not available - HuggingFace token required")

        config = config or DiarizationConfig.get_balanced_config()
        start_time = time.time()

        logger.info(f"👥 Starting diarization for transcription with {len(transcription_result.segments)} segments")

        try:
            if progress_callback:
                progress_callback("diarizing_audio_overall", 0.05) # Initial overall progress

            # === STEP 1: PERFORM DIARIZATION ===
            diarization_start_ts = time.time()
            
            # This lambda scales the progress from diarize_audio (0.0-1.0) to a portion of the overall progress (e.g., 0.1 to 0.7)
            def engine_progress_adapter(stage: str, engine_progress_value: float):
                if progress_callback:
                    # Map engine_progress_value (0-1) to a range, e.g., 0.1 (start of engine work) to 0.7 (end of engine work)
                    current_overall_progress = 0.1 + engine_progress_value * 0.6 
                    progress_callback(stage, current_overall_progress)

            raw_diarization_output = await self.engine.diarize_audio(
                audio=audio,
                sample_rate=sample_rate,
                config=config,
                progress_callback=engine_progress_adapter if progress_callback else None
            )
            
            diarization_time = time.time() - diarization_start_ts
            model_load_time = raw_diarization_output.get("pipeline_load_time", 0.0) # Get from engine result

            if progress_callback:
                progress_callback("assigning_speakers", 0.7)

            # === STEP 2: ASSIGN SPEAKERS TO TRANSCRIPTION ===
            assignment_start_ts = time.time()

            trans_segments_as_dicts = []
            for segment in transcription_result.segments:
                trans_segments_as_dicts.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    # Pass words if available, for richer SpeakerSegment later
                    "words": segment.words if hasattr(segment, 'words') else None, 
                    "confidence": segment.avg_logprob if hasattr(segment, 'avg_logprob') else None # Example: if confidence is available
                })
            
            # raw_diarization_output["segments"] contains the speaker turns from the engine
            assigned_segments_dicts = await asyncio.to_thread(
                self.assigner.assign_speakers_to_transcription,
                trans_segments_as_dicts,
                raw_diarization_output.get("segments", []),
                config 
            )
            
            assignment_time = time.time() - assignment_start_ts

            if progress_callback:
                progress_callback("processing_final_results", 0.9)

            # === STEP 3: PROCESS RESULTS INTO DiarizationResult DATACLASS ===
            result = self._create_diarization_result(
                assigned_segments_dicts,
                transcription_result.duration, # Total audio duration from transcription
                config,
                time.time() - start_time, # Total processing time for this service call
                model_load_time,
                diarization_time,
                assignment_time
            )

            if progress_callback:
                progress_callback("completed", 1.0)

            logger.info(
                f"✅ Diarization completed: {result.speaker_count} speakers, "
                f"{len(result.segments)} segments in {result.processing_time:.2f}s "
                f"(ModelLoad: {model_load_time:.2f}s, Diarize: {diarization_time:.2f}s, Assign: {assignment_time:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}", exc_info=True)
            raise

    async def diarize_alignment_result(
        self,
        alignment_result: AlignmentResult,
        audio: np.ndarray,
        sample_rate: int,
        config: Optional[DiarizationConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> DiarizationResult:
        """
        Add speaker diarization to alignment result.
        (This function converts AlignmentResult to a mock TranscriptionResult and calls the main method)
        """
        if not self.is_available:
            raise ValueError("Diarization service not available")

        from services.transcription_service import TranscriptionSegment as WhisperTranscriptionSegment # Avoid name clash if imported directly
        
        transcription_segments_for_diar = []
        total_aligned_duration = 0.0
        for aligned_seg in alignment_result.segments:
            # Convert AlignedSegment words to the dict format expected by TranscriptionSegment
            words_list = [{
                "start": word.start, "end": word.end, "word": word.word, 
                "score": word.score if hasattr(word, 'score') else (word.probability if hasattr(word, 'probability') else None) # whisperx alignment uses 'score'
            } for word in aligned_seg.words] if aligned_seg.words else None

            trans_seg = WhisperTranscriptionSegment(
                start=aligned_seg.start,
                end=aligned_seg.end,
                text=aligned_seg.text,
                words=words_list
                # Other fields like avg_logprob, no_speech_prob could be added if available/relevant
            )
            transcription_segments_for_diar.append(trans_seg)
            total_aligned_duration += (aligned_seg.end - aligned_seg.start)
        
        # Create a mock TranscriptionResult
        # Note: transcription_result.duration is used by _create_diarization_result
        # It should ideally be the total audio duration, not just sum of segment durations.
        # For now, using sum of aligned segments as a proxy if total audio duration isn't passed.
        # If alignment_result has a total_duration field for the original audio, use that.
        # Assuming alignment_result.duration refers to the total duration of the processed audio.
        audio_total_duration = alignment_result.duration if hasattr(alignment_result, 'duration') and alignment_result.duration > 0 else total_aligned_duration


        mock_transcription_result = TranscriptionResult(
            segments=transcription_segments_for_diar,
            language=alignment_result.language,
            duration=audio_total_duration, 
            # model_name=alignment_result.model_name or "aligned_model", # Assuming AlignmentResult has model_name
            # config=None, # TranscriptionConfig, not DiarizationConfig
            processing_time=0.0 # Mocked, as this is for the transcription part
        )

        return await self.diarize_transcription_result(
            mock_transcription_result, audio, sample_rate, config, progress_callback
        )

    def _create_diarization_result(
        self,
        assigned_segments_dicts: List[Dict[str, Any]], # These are transcription segments with 'speaker' key
        total_audio_duration: float,
        config: DiarizationConfig,
        overall_processing_time: float,
        model_load_time: Optional[float],
        diarization_engine_time: Optional[float],
        assignment_time: Optional[float]
    ) -> DiarizationResult:
        """Helper to create the final DiarizationResult object"""

        unique_speaker_ids = sorted(list(set(
            seg.get("speaker", "SPEAKER_UNKNOWN") for seg in assigned_segments_dicts
        )))

        speaker_labels_map = self.labeler.generate_labels(
            unique_speaker_ids,
            config.speaker_label_mode,
            config.custom_speaker_names
        )

        final_speaker_segments = []
        for seg_dict in assigned_segments_dicts:
            speaker_id = seg_dict.get("speaker", "SPEAKER_UNKNOWN")
            # Ensure words are correctly passed if they exist on seg_dict
            words_data = seg_dict.get("words") if isinstance(seg_dict.get("words"), list) else None

            speaker_segment = SpeakerSegment(
                speaker_id=speaker_id, # This is the raw ID like SPEAKER_00
                start=float(seg_dict.get("start", 0)),
                end=float(seg_dict.get("end", 0)),
                text=str(seg_dict.get("text", "")),
                # confidence=seg_dict.get("confidence"), # From trans_segments_as_dicts
                words=words_data # From trans_segments_as_dicts
            )
            final_speaker_segments.append(speaker_segment)
        
        # Sort segments by start time
        final_speaker_segments.sort(key=lambda s: s.start)

        speaker_profiles_list = []
        for speaker_id_val in unique_speaker_ids:
            segments_for_this_speaker = [
                s for s in final_speaker_segments if s.speaker_id == speaker_id_val
            ]
            
            if segments_for_this_speaker:
                total_speak_time = sum(s.duration() for s in segments_for_this_speaker)
                num_segments = len(segments_for_this_speaker)
                avg_seg_duration = total_speak_time / num_segments if num_segments > 0 else 0.0
                
                profile = SpeakerProfile(
                    speaker_id=speaker_id_val,
                    speaker_label=speaker_labels_map.get(speaker_id_val, speaker_id_val),
                    total_speaking_time=total_speak_time,
                    segment_count=num_segments,
                    average_segment_duration=avg_seg_duration
                    # Add confidence if available/meaningful at speaker level
                )
                speaker_profiles_list.append(profile)
        
        # Sort speakers by total speaking time (most active first) or by label
        speaker_profiles_list.sort(key=lambda p: p.total_speaking_time, reverse=True)

        return DiarizationResult(
            segments=final_speaker_segments,
            speakers=speaker_profiles_list,
            total_duration=total_audio_duration,
            config=config,
            processing_time=overall_processing_time,
            model_load_time=model_load_time,
            diarization_time=diarization_engine_time,
            assignment_time=assignment_time
        )
    
    def get_diarization_recommendations(
        self,
        audio_duration: float,
        expected_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get diarization recommendations based on audio characteristics.
        (Implementation assumed to be correct as per snippet)
        """
        recommendations = {
            "service_available": self.is_available,
            "hardware_device": self.hardware_caps.device
        }

        if not self.is_available:
            recommendations["message"] = "Diarization not available - HuggingFace token required"
            return recommendations

        if audio_duration < 300:  # < 5 minutes
            recommendations["recommended_quality"] = DiarizationQuality.ACCURATE.value # Return value
            recommendations["reason"] = "Short audio allows for accurate diarization"
        elif audio_duration < 1800:  # < 30 minutes
            recommendations["recommended_quality"] = DiarizationQuality.BALANCED.value
            recommendations["reason"] = "Medium audio benefits from balanced approach"
        else:  # > 30 minutes
            recommendations["recommended_quality"] = DiarizationQuality.FAST.value
            recommendations["reason"] = "Long audio requires fast processing"

        if expected_speakers:
            max_s = min(expected_speakers + 2, 12)  # Allow some flexibility, cap at a reasonable number
            min_s = max(1, expected_speakers -1) if expected_speakers > 1 else 1
        else:
            if audio_duration < 600:  # < 10 minutes
                max_s = 6
                min_s = None # Let the model decide if not many speakers expected
            else:
                max_s = 10
                min_s = 2 # Assume at least 2 for longer audio

        recommendations["recommended_max_speakers"] = max_s
        recommendations["recommended_min_speakers"] = min_s
        
        # Estimate processing time more realistically
        # These are very rough estimates and highly dependent on hardware and specific audio
        quality_factor = {
            DiarizationQuality.FAST.value: 0.05,  # e.g., 3 mins for 1 hour audio
            DiarizationQuality.BALANCED.value: 0.1, # e.g., 6 mins for 1 hour audio
            DiarizationQuality.ACCURATE.value: 0.2  # e.g., 12 mins for 1 hour audio
        }.get(recommendations["recommended_quality"], 0.1)

        # Hardware factor (very simplified)
        hw_factor = 1.0 if self.hardware_caps.device == "cuda" else 3.0 
        
        estimated_processing_seconds = audio_duration * quality_factor * hw_factor
        recommendations["estimated_processing_time_minutes"] = round(max(0.5, estimated_processing_seconds / 60), 1)
        
        return recommendations

    def cleanup_resources(self):
        """Clean up diarization resources"""
        self.engine.cleanup_pipeline()


# Convenience functions
async def diarize_transcription_fast(
    transcription_result: TranscriptionResult,
    audio: np.ndarray,
    sample_rate: int,
    diarization_service: DiarizationService,
    max_speakers: int = 6
) -> DiarizationResult:
    """Fast diarization with basic speaker detection"""
    config = DiarizationConfig.get_fast_config(max_speakers)
    return await diarization_service.diarize_transcription_result(
        transcription_result, audio, sample_rate, config
    )


async def diarize_transcription_balanced(
    transcription_result: TranscriptionResult,
    audio: np.ndarray,
    sample_rate: int,
    diarization_service: DiarizationService,
    max_speakers: int = 8
) -> DiarizationResult:
    """Balanced diarization with good accuracy"""
    config = DiarizationConfig.get_balanced_config(max_speakers)
    return await diarization_service.diarize_transcription_result(
        transcription_result, audio, sample_rate, config
    )


async def diarize_meeting_transcription( # This is the correct, single definition
    transcription_result: TranscriptionResult,
    audio: np.ndarray,
    sample_rate: int,
    diarization_service: DiarizationService,
    expected_speakers: int = 6
) -> DiarizationResult:
    """Diarization optimized for meeting recordings"""
    config = DiarizationConfig.get_meeting_config(expected_speakers)
    return await diarization_service.diarize_transcription_result(
        transcription_result, audio, sample_rate, config
    )

