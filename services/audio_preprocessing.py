# services/audio_preprocessing.py
"""
Audio Preprocessing Service

Erweiterte Audio-Verarbeitung für ML-Pipeline-Vorbereitung.
Wie ein Audio-Ingenieur, der das Signal perfekt für die
nachgelagerte Verarbeitung aufbereitet.
"""

import logging
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import librosa
from scipy import signal
import noisereduce as nr

logger = logging.getLogger(__name__)


class NoiseReductionMethod(Enum):
    """Verfügbare Noise-Reduction-Methoden"""
    NONE = "none"
    SPECTRAL_GATING = "spectral_gating"
    WIENER_FILTER = "wiener_filter"  # Implementierung fehlt noch
    STATIONARY = "stationary"


class NormalizationMethod(Enum):
    """Audio-Normalisierungs-Methoden"""
    NONE = "none"
    PEAK = "peak"            # Peak normalization zu [-1, 1]
    RMS = "rms"              # RMS-basierte Normalisierung
    LUFS = "lufs"            # LUFS (Loudness Units relative to Full Scale)


@dataclass
class PreprocessingConfig:
    """Konfiguration für Audio-Preprocessing"""

    # Noise Reduction
    noise_reduction: NoiseReductionMethod = NoiseReductionMethod.SPECTRAL_GATING
    noise_reduce_strength: float = 0.5  # 0.0 = off, 1.0 = aggressive

    # Normalization
    normalization: NormalizationMethod = NormalizationMethod.RMS
    target_rms_db: float = -20.0  # Target RMS level in dB

    # Filtering
    apply_highpass: bool = True
    highpass_cutoff: float = 80.0  # Hz - removes low-frequency noise
    apply_lowpass: bool = False
    lowpass_cutoff: float = 8000.0  # Hz

    # Silence Detection & Trimming
    trim_silence: bool = True
    silence_threshold_db: float = -40.0  # dB below peak
    silence_margin_seconds: float = 0.1  # Keep margin around speech

    # Advanced Processing
    apply_compander: bool = False
    compander_ratio: float = 3.0  # Compression ratio

    # Voice Activity Detection preprocessing
    vad_preprocessing: bool = True # Ob VAD-spezifische Schritte angewendet werden sollen
    vad_frame_duration: float = 0.025  # 25ms frames (für interne VAD, falls genutzt)

    @classmethod
    def get_speech_optimized(cls) -> 'PreprocessingConfig':
        """Gibt Speech-optimierte Konfiguration zurück"""
        return cls(
            noise_reduction=NoiseReductionMethod.SPECTRAL_GATING,
            noise_reduce_strength=0.7,
            normalization=NormalizationMethod.RMS,
            target_rms_db=-18.0,
            apply_highpass=True,
            highpass_cutoff=100.0,  # Remove more low-frequency noise for speech
            trim_silence=True,
            silence_threshold_db=-35.0,
            vad_preprocessing=True
        )

    @classmethod
    def get_minimal(cls) -> 'PreprocessingConfig':
        """Minimale Verarbeitung für High-Quality-Audio"""
        return cls(
            noise_reduction=NoiseReductionMethod.NONE,
            normalization=NormalizationMethod.PEAK,
            apply_highpass=False,
            trim_silence=False,
            vad_preprocessing=False
        )


@dataclass
class PreprocessingResult:
    """Ergebnis der Audio-Vorverarbeitung"""
    processed_audio: np.ndarray
    original_duration: float
    sample_rate: int
    processed_duration: float = 0.0  # KORREKTUR: Standardwert hinzugefügt

    # Processing statistics
    noise_reduction_applied: bool = False
    silence_trimmed_seconds: float = 0.0
    peak_before: float = 0.0
    peak_after: float = 0.0
    rms_before_db: float = 0.0
    rms_after_db: float = 0.0

    # Performance metrics
    processing_time_seconds: float = 0.0

    def get_processing_summary(self) -> Dict[str, Any]:
        """Gibt Zusammenfassung der Verarbeitung zurück"""
        return {
            "duration_change_seconds": self.processed_duration - self.original_duration,
            "silence_removed_seconds": self.silence_trimmed_seconds,
            "peak_change_abs": self.peak_after - self.peak_before,
            "rms_change_db": self.rms_after_db - self.rms_before_db,
            "processing_time_seconds": self.processing_time_seconds,
            "noise_reduction_applied": self.noise_reduction_applied
        }


class NoiseReducer:
    """Noise Reduction Implementierungen"""

    @staticmethod
    def spectral_gating(
        audio: np.ndarray,
        sr: int,
        strength: float = 0.5
    ) -> Tuple[np.ndarray, bool]:
        """
        Spectral Gating Noise Reduction.
        """
        if strength <= 0.0:
            return audio, False

        try:
            # Verwende noisereduce library
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=False,  # Non-stationary noise reduction
                prop_decrease=strength,
                # Weitere Parameter können hier hinzugefügt werden, falls nötig
                # z.B. n_fft, hop_length, freq_mask_smooth_hz, time_mask_smooth_ms
            )
            logger.debug(f"✅ Spectral gating applied (strength: {strength})")
            return reduced_audio, True
        except Exception as e:
            logger.warning(f"⚠️ Spectral gating failed: {e}", exc_info=True)
            return audio, False

    @staticmethod
    def stationary_noise_reduction(
        audio: np.ndarray,
        sr: int,
        strength: float = 0.5
    ) -> Tuple[np.ndarray, bool]:
        """Stationary Noise Reduction"""
        if strength <= 0.0:
            return audio, False
        try:
            reduced_audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,  # Stationary noise reduction
                prop_decrease=strength
            )
            logger.debug(f"✅ Stationary noise reduction applied (strength: {strength})")
            return reduced_audio, True
        except Exception as e:
            logger.warning(f"⚠️ Stationary noise reduction failed: {e}", exc_info=True)
            return audio, False

    # Hier könnte die Implementierung für WIENER_FILTER folgen, falls benötigt


class AudioNormalizer:
    """Audio-Normalisierungs-Implementierungen"""

    @staticmethod
    def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Peak-basierte Normalisierung."""
        current_peak = np.max(np.abs(audio))
        if current_peak > 1e-6: # Avoid division by zero or very small numbers
            gain = target_peak / current_peak
            return audio * gain
        return audio

    @staticmethod
    def rms_normalize(audio: np.ndarray, target_rms_db: float = -20.0) -> np.ndarray:
        """RMS-basierte Normalisierung."""
        current_rms_linear = np.sqrt(np.mean(audio**2))
        if current_rms_linear < 1e-9: # Audio is practically silent
            logger.debug("Audio is silent, RMS normalization skipped.")
            return audio

        target_rms_linear = 10**(target_rms_db / 20.0)
        gain = target_rms_linear / current_rms_linear

        # Begrenze Gain um Clipping zu vermeiden
        # Es ist besser, das normalisierte Audio zu clippen, als den Gain zu stark zu begrenzen,
        # da sonst das RMS-Ziel nicht erreicht wird.
        normalized_audio = audio * gain
        
        # Optional: Soft Clipping oder Hard Clipping, falls Peaks > 1.0 auftreten
        # Hard Clipping:
        # if np.max(np.abs(normalized_audio)) > 1.0:
        #     logger.warning("Clipping occurred during RMS normalization. Applying hard clip.")
        #     normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
            
        return normalized_audio

    @staticmethod
    def lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
        """LUFS-basierte Normalisierung (vereinfacht)."""
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr) # K_WEIGHTING, block_size=0.400
            current_lufs = meter.integrated_loudness(audio)

            if not np.isfinite(current_lufs) or current_lufs < -70.0: # -70 LUFS is effectively silent
                logger.warning(f"⚠️ Could not measure LUFS reliably (value: {current_lufs}), using RMS fallback or returning original.")
                # Fallback zu RMS oder Original-Audio zurückgeben, wenn RMS auch fehlschlägt
                return AudioNormalizer.rms_normalize(audio, target_lufs + 3.0) # Empirischer Offset

            gain_db = target_lufs - current_lufs
            gain_linear = 10**(gain_db / 20.0)

            normalized_audio = audio * gain_linear
            
            # Optional: Clipping-Behandlung
            # if np.max(np.abs(normalized_audio)) > 1.0:
            #     logger.warning("Clipping occurred during LUFS normalization. Applying hard clip.")
            #     normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
                
            return normalized_audio

        except ImportError:
            logger.warning("⚠️ pyloudnorm not available, using RMS fallback for LUFS normalization.")
            return AudioNormalizer.rms_normalize(audio, target_lufs + 3.0) # Empirischer Offset
        except Exception as e:
            logger.warning(f"⚠️ LUFS normalization failed: {e}, using RMS fallback.", exc_info=True)
            return AudioNormalizer.rms_normalize(audio, target_lufs + 3.0)


class AudioFilter:
    """Audio-Filtering-Implementierungen"""

    @staticmethod
    def highpass_filter(
        audio: np.ndarray,
        sr: int,
        cutoff_freq: float,
        order: int = 5
    ) -> np.ndarray:
        """Hochpass-Filter zur Entfernung tiefer Frequenzen."""
        try:
            nyquist = 0.5 * sr
            if cutoff_freq >= nyquist:
                logger.warning(f"Highpass cutoff frequency ({cutoff_freq}Hz) is too high for sample rate ({sr}Hz). Skipping filter.")
                return audio
            if cutoff_freq <= 0:
                logger.debug("Highpass cutoff frequency is zero or negative. Skipping filter.")
                return audio
                
            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio)
            logger.debug(f"✅ Highpass filter applied: {cutoff_freq}Hz")
            return filtered_audio
        except Exception as e:
            logger.warning(f"⚠️ Highpass filter failed: {e}", exc_info=True)
            return audio

    @staticmethod
    def lowpass_filter(
        audio: np.ndarray,
        sr: int,
        cutoff_freq: float,
        order: int = 5
    ) -> np.ndarray:
        """Tiefpass-Filter zur Entfernung hoher Frequenzen."""
        try:
            nyquist = 0.5 * sr
            if cutoff_freq >= nyquist:
                logger.warning(f"Lowpass cutoff frequency ({cutoff_freq}Hz) is too high for sample rate ({sr}Hz). Skipping filter.")
                return audio
            if cutoff_freq <= 0: # Should not happen for lowpass, but good to check
                return audio

            normalized_cutoff = cutoff_freq / nyquist
            b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
            filtered_audio = signal.filtfilt(b, a, audio)
            logger.debug(f"✅ Lowpass filter applied: {cutoff_freq}Hz")
            return filtered_audio
        except Exception as e:
            logger.warning(f"⚠️ Lowpass filter failed: {e}", exc_info=True)
            return audio


class SilenceTrimmer:
    """Stille-Erkennung und -Entfernung"""

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sr: int,
        threshold_db: float = -40.0, # dBFS, relative to peak
        min_silence_len_ms: int = 500, # Minimum duration of silence to be considered for trimming (not used by librosa.effects.trim)
        padding_ms: int = 100 # Keep some silence at ends
    ) -> Tuple[np.ndarray, float]:
        """
        Entfernt Stille am Anfang und Ende des Audios mit Librosa.
        """
        try:
            # Librosa's trim function uses top_db which is threshold below the peak
            trimmed_audio, index = librosa.effects.trim(audio, top_db=-threshold_db) # Note: librosa uses positive dB for threshold
            
            original_duration = len(audio) / sr
            trimmed_duration_val = len(trimmed_audio) / sr
            removed_seconds = original_duration - trimmed_duration_val

            if removed_seconds > 0.01: # Only log if significant trimming occurred
                logger.debug(f"✅ Silence trimmed (librosa): {removed_seconds:.2f}s removed")
            return trimmed_audio, removed_seconds
        except Exception as e:
            logger.warning(f"⚠️ Silence trimming with librosa failed: {e}", exc_info=True)
            # Fallback zu einer einfacheren Methode oder Original zurückgeben
            return audio, 0.0


class VoiceActivityDetector:
    """Voice Activity Detection für Preprocessing"""

    @staticmethod
    def detect_voice_segments(
        audio: np.ndarray,
        sr: int,
        frame_duration_ms: float = 30, # ms
        hop_duration_ms: float = 10,   # ms
        vad_threshold: float = 0.5,    # Schwellwert für VAD-Modell
        min_speech_duration_ms: float = 250, # ms
        min_silence_duration_ms: float = 100, # ms
        padding_duration_ms: float = 100 # ms
    ) -> List[Tuple[float, float]]:
        """
        Erkennt Sprach-Segmente im Audio mit einem einfachen Energie-basierten VAD
        oder idealerweise mit einem robusteren Modell (z.B. Silero VAD).
        Diese Implementierung ist eine einfache Energie-basierte Version.
        """
        try:
            # Für eine robuste VAD wird Silero VAD empfohlen.
            # Beispiel für Silero VAD (erfordert Installation und Modell-Download):
            # model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            # (get_speech_timestamps, _, read_audio, *_) = utils
            # speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=sr,
            #                                           threshold=vad_threshold,
            #                                           min_speech_duration_ms=min_speech_duration_ms,
            #                                           min_silence_duration_ms=min_silence_duration_ms,
            #                                           window_size_samples=... , # calculate from frame_duration_ms
            #                                           speech_pad_ms=padding_duration_ms)
            # return [(ts['start']/sr, ts['end']/sr) for ts in speech_timestamps]

            # Einfache Energie-basierte VAD als Fallback/Beispiel:
            frame_length = int(sr * frame_duration_ms / 1000)
            hop_length = int(sr * hop_duration_ms / 1000)
            rms_energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Normalisiere Energie und setze Schwellwert
            if np.max(rms_energy) > 1e-6:
                 rms_energy_norm = rms_energy / np.max(rms_energy)
            else:
                rms_energy_norm = rms_energy # Avoid division by zero if silent

            speech_frames = rms_energy_norm > vad_threshold # vad_threshold hier als relativer Schwellwert

            # Konvertiere Frames zu Zeitstempeln und merge
            speech_timestamps_frames = []
            is_speech = False
            start_frame = 0
            for i, frame_is_speech in enumerate(speech_frames):
                if frame_is_speech and not is_speech:
                    start_frame = i
                    is_speech = True
                elif not frame_is_speech and is_speech:
                    # TODO: Implement merging based on min_speech_duration_ms etc.
                    # This requires more complex logic than a simple frame-by-frame conversion
                    speech_timestamps_frames.append({'start': start_frame * hop_length, 'end': i * hop_length})
                    is_speech = False
            if is_speech: # Letztes Segment
                speech_timestamps_frames.append({'start': start_frame * hop_length, 'end': len(speech_frames) * hop_length})

            # Konvertiere sample-basierte Zeitstempel zu Sekunden
            segments = [(ts['start']/sr, ts['end']/sr) for ts in speech_timestamps_frames]
            
            # Hier wäre eine Post-Processing-Logik für min_speech_duration etc. nötig
            # Für eine robuste Lösung wird Silero VAD oder eine ähnliche Bibliothek empfohlen.

            if not segments: # Fallback: ganzes Audio als Sprache, wenn nichts erkannt wird
                 logger.debug("VAD (energiebasiert) hat keine Sprachsegmente erkannt, gibt gesamtes Audio zurück.")
                 return [(0.0, len(audio) / sr)]

            logger.debug(f"✅ VAD (energiebasiert) detected {len(segments)} voice segments")
            return segments

        except Exception as e:
            logger.warning(f"⚠️ VAD failed: {e}", exc_info=True)
            return [(0.0, len(audio) / sr)]  # Fallback: ganzes Audio als Sprache


class AudioPreprocessor:
    """
    Haupt-Audio-Preprocessing-Service.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig.get_speech_optimized() # Default zu speech_optimized

        self.noise_reducer = NoiseReducer()
        self.normalizer = AudioNormalizer()
        self.filter = AudioFilter()
        self.silence_trimmer = SilenceTrimmer()
        self.vad = VoiceActivityDetector()

    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        config_override: Optional[PreprocessingConfig] = None
    ) -> PreprocessingResult:
        """
        Führt vollständige Audio-Vorverarbeitung durch.
        """
        config = config_override or self.config
        start_time = time.time()

        logger.info(f"🎛️ Starting audio preprocessing: {len(audio)/sr:.2f}s @ {sr}Hz with config: {config.noise_reduction.value}, norm: {config.normalization.value}")

        original_audio_duration = len(audio) / sr
        initial_peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
        initial_rms_val = np.sqrt(np.mean(audio**2)) if len(audio) > 0 else 0.0
        initial_rms_db = 20 * np.log10(initial_rms_val) if initial_rms_val > 1e-9 else -120.0 # Avoid log(0)

        current_audio = audio.copy() # Arbeite immer mit einer Kopie

        # 1. Noise Reduction
        noise_reduction_applied_flag = False
        if config.noise_reduction != NoiseReductionMethod.NONE:
            current_audio, noise_reduction_applied_flag = self._apply_noise_reduction(
                current_audio, sr, config
            )

        # 2. Filtering
        if config.apply_highpass:
            current_audio = self.filter.highpass_filter(
                current_audio, sr, config.highpass_cutoff
            )
        if config.apply_lowpass:
            current_audio = self.filter.lowpass_filter(
                current_audio, sr, config.lowpass_cutoff
            )

        # 3. Silence Trimming
        silence_trimmed_seconds_val = 0.0
        if config.trim_silence:
            current_audio, silence_trimmed_seconds_val = self.silence_trimmer.trim_silence(
                current_audio, sr, config.silence_threshold_db, padding_ms=int(config.silence_margin_seconds * 1000)
            )
        
        # 4. Normalization
        if config.normalization != NormalizationMethod.NONE:
            current_audio = self._apply_normalization(current_audio, sr, config)

        # 5. Advanced Processing (Optional) - Compander ist hier sehr rudimentär
        if config.apply_compander:
            current_audio = self._apply_compander(current_audio, config.compander_ratio)


        # Finale Metriken nach der Verarbeitung
        final_processed_duration = len(current_audio) / sr if len(current_audio) > 0 else 0.0
        final_peak = np.max(np.abs(current_audio)) if len(current_audio) > 0 else 0.0
        final_rms_val = np.sqrt(np.mean(current_audio**2)) if len(current_audio) > 0 else 0.0
        final_rms_db = 20 * np.log10(final_rms_val) if final_rms_val > 1e-9 else -120.0


        result = PreprocessingResult(
            processed_audio=current_audio,
            original_duration=original_audio_duration,
            processed_duration=final_processed_duration, # Jetzt bekannt
            sample_rate=sr,
            noise_reduction_applied=noise_reduction_applied_flag,
            silence_trimmed_seconds=silence_trimmed_seconds_val,
            peak_before=initial_peak,
            peak_after=final_peak,
            rms_before_db=initial_rms_db,
            rms_after_db=final_rms_db,
            processing_time_seconds = time.time() - start_time
        )

        logger.info(
            f"✅ Audio preprocessing complete: "
            f"{result.original_duration:.2f}s → {result.processed_duration:.2f}s "
            f"in {result.processing_time_seconds:.2f}s"
        )
        return result

    def _apply_noise_reduction(
        self,
        audio: np.ndarray,
        sr: int,
        config: PreprocessingConfig
    ) -> Tuple[np.ndarray, bool]:
        """Anwenden der konfigurierten Noise Reduction"""
        if config.noise_reduction == NoiseReductionMethod.SPECTRAL_GATING:
            return self.noise_reducer.spectral_gating(
                audio, sr, config.noise_reduce_strength
            )
        elif config.noise_reduction == NoiseReductionMethod.STATIONARY:
            return self.noise_reducer.stationary_noise_reduction(
                audio, sr, config.noise_reduce_strength
            )
        # elif config.noise_reduction == NoiseReductionMethod.WIENER_FILTER:
        # Implement Wiener Filter if needed
        # return self.noise_reducer.wiener_filter(audio, sr), True
        return audio, False

    def _apply_normalization(
        self,
        audio: np.ndarray,
        sr: int,
        config: PreprocessingConfig
    ) -> np.ndarray:
        """Anwenden der konfigurierten Normalisierung"""
        if config.normalization == NormalizationMethod.PEAK:
            return self.normalizer.peak_normalize(audio)
        elif config.normalization == NormalizationMethod.RMS:
            return self.normalizer.rms_normalize(audio, config.target_rms_db)
        elif config.normalization == NormalizationMethod.LUFS:
            # LUFS Normalisierung erwartet Zielwert in LUFS, nicht dB RMS
            # Hier könnte eine separate target_lufs Konfiguration sinnvoll sein.
            # Annahme: target_rms_db wird als Näherung für target_lufs verwendet, wenn LUFS gewählt wird.
            # Ein typischer Wert für LUFS ist -23.0 oder -18.0 für Sprache.
            target_lufs_value = config.target_rms_db # Oder eine dedizierte config.target_lufs
            if config.target_rms_db > -10: # RMS dB Werte sind oft niedriger als LUFS Werte
                 logger.warning(f"Target RMS dB ({config.target_rms_db}) seems high for LUFS. Consider using a target around -18 to -23 LUFS.")
            return self.normalizer.lufs_normalize(audio, sr, target_lufs_value)
        return audio

    def _apply_compander(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Einfacher Kompressor/Expander (sehr rudimentär)"""
        try:
            # Vereinfachter Compander (nur Kompression)
            threshold = 0.1 # Schwellwert für Kompression
            # Audio muss im Bereich [-1, 1] sein für diese Logik
            
            # Sicherstellen, dass Audio normalisiert ist, bevor Compander angewendet wird
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))

            compressed_audio = audio.copy()
            above_threshold_indices = np.abs(audio) > threshold
            
            # Kompression für Signale über dem Schwellwert
            compressed_audio[above_threshold_indices] = \
                np.sign(audio[above_threshold_indices]) * \
                (threshold + (np.abs(audio[above_threshold_indices]) - threshold) / ratio)

            logger.debug(f"✅ Rudimentary Compander applied (ratio: {ratio})")
            return compressed_audio
        except Exception as e:
            logger.warning(f"⚠️ Compander failed: {e}", exc_info=True)
            return audio

    def get_voice_segments(self, audio: np.ndarray, sr: int, config: Optional[PreprocessingConfig] = None) -> List[Tuple[float, float]]:
        """Erkennt Sprach-Segmente im Audio, verwendet VAD-Einstellungen aus der Config."""
        current_config = config or self.config
        # Hier könnten spezifische VAD-Parameter aus current_config an self.vad.detect_voice_segments übergeben werden
        return self.vad.detect_voice_segments(
            audio, 
            sr, 
            frame_duration_ms=current_config.vad_frame_duration * 1000, # Konvertiere zu ms
            # Weitere Parameter wie vad_threshold etc. könnten hier aus current_config kommen
            )


# Default Preprocessor Instances
default_preprocessor = AudioPreprocessor() # Verwendet speech_optimized per Default
speech_optimized_preprocessor = AudioPreprocessor(PreprocessingConfig.get_speech_optimized())
minimal_preprocessor = AudioPreprocessor(PreprocessingConfig.get_minimal())


# Convenience Functions
def preprocess_audio(audio: np.ndarray, sr: int, config: Optional[PreprocessingConfig] = None) -> PreprocessingResult:
    """Convenience function für Audio-Preprocessing"""
    # Wenn keine spezifische Konfig übergeben wird, nimmt der default_preprocessor seine Standardkonfig.
    # Wenn eine config übergeben wird, wird diese verwendet.
    return default_preprocessor.preprocess_audio(audio, sr, config_override=config)


def preprocess_for_speech(audio: np.ndarray, sr: int) -> PreprocessingResult:
    """Convenience function für Speech-optimiertes Preprocessing"""
    return speech_optimized_preprocessor.preprocess_audio(audio, sr)


def preprocess_minimal(audio: np.ndarray, sr: int) -> PreprocessingResult:
    """Convenience function für minimales Preprocessing"""
    return minimal_preprocessor.preprocess_audio(audio, sr)