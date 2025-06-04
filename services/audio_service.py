# services/audio_service.py
"""
Audio Processing Service

Kapselt alle Audio-bezogenen Operationen in einen sauberen Service.
Wie ein Audio-Techniker, der genau weiß, welche Kabel wohin gehören
und warum der Sound manchmal kratzt.
"""

import asyncio
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Unterstützte Audio-Formate"""
    WAV = "wav"
    MP3 = "mp3"
    MP4 = "mp4"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"


@dataclass
class AudioMetadata:
    """Metadaten einer Audio-Datei"""
    duration_seconds: float
    sample_rate: int
    channels: int
    file_size_bytes: int
    format: str
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    is_valid: bool = True
    validation_message: str = "OK"


class AudioValidationError(Exception):
    """Fehler bei Audio-Validierung"""
    pass


class AudioLoadingStrategy:
    """Base class für verschiedene Audio-Loading-Strategien"""
    
    def load_audio(
        self, 
        file_path: Union[str, Path], 
        sr: int = 16000, 
        mono: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Lädt Audio-Datei"""
        raise NotImplementedError
    
    def get_strategy_name(self) -> str:
        """Name der Strategie"""
        raise NotImplementedError


class SoundFileStrategy(AudioLoadingStrategy):
    """Lädt Audio mit SoundFile (WAV, FLAC, OGG)"""
    
    def load_audio(
        self, 
        file_path: Union[str, Path], 
        sr: int = 16000, 
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> np.ndarray:
        """Lädt Audio mit SoundFile"""
        file_path = str(file_path)
        logger.debug(f"🎵 Loading with SoundFile: {file_path}")
        
        # Parameter für Partial-Loading
        sf_kwargs = {}
        if offset > 0:
            sf_kwargs['start'] = int(offset * sr)  # Approximation
        if duration:
            sf_kwargs['frames'] = int(duration * sr)  # Approximation
        
        # Audio laden
        audio, original_sr = sf.read(file_path, **sf_kwargs)
        logger.debug(f"✅ SoundFile loaded: {audio.shape}, {original_sr}Hz")
        
        return self._post_process_audio(audio, original_sr, sr, mono)
    
    def _post_process_audio(
        self, 
        audio: np.ndarray, 
        original_sr: int, 
        target_sr: int, 
        mono: bool
    ) -> np.ndarray:
        """Post-Processing: Mono-Conversion und Resampling"""
        
        # Zu Mono konvertieren
        if mono and len(audio.shape) > 1:
            if audio.shape[1] == 2:  # Stereo
                audio = np.mean(audio, axis=1)
            elif audio.shape[0] == 2:  # Transposed Stereo
                audio = np.mean(audio, axis=0)
        
        # Resampling falls nötig
        if original_sr != target_sr:
            logger.debug(f"🔄 Resampling: {original_sr}Hz -> {target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        
        # Typ-Konvertierung und Normalisierung
        audio = audio.astype(np.float32)
        
        # Clipping-Behandlung
        if np.max(np.abs(audio)) > 1.0:
            logger.warning("⚠️ Audio clipping detected - normalizing...")
            audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def get_strategy_name(self) -> str:
        return "soundfile"


class LibrosaStrategy(AudioLoadingStrategy):
    """Lädt Audio mit Librosa (robuster, alle Formate)"""
    
    def load_audio(
        self, 
        file_path: Union[str, Path], 
        sr: int = 16000, 
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> np.ndarray:
        """Lädt Audio mit Librosa"""
        file_path = str(file_path)
        logger.debug(f"🎵 Loading with Librosa: {file_path}")
        
        librosa_kwargs = {'sr': sr, 'mono': mono}
        if offset > 0:
            librosa_kwargs['offset'] = offset
        if duration:
            librosa_kwargs['duration'] = duration
        
        audio, loaded_sr = librosa.load(file_path, **librosa_kwargs)
        logger.debug(f"✅ Librosa loaded: {audio.shape}, {loaded_sr}Hz")
        
        return audio.astype(np.float32)
    
    def get_strategy_name(self) -> str:
        return "librosa"


class PyDubStrategy(AudioLoadingStrategy):
    """Lädt Audio mit PyDub (MP3, MP4, M4A usw.)"""

    def __init__(self):
        # Die PyDub-Importprüfung kann hier erfolgen, um frühzeitig zu scheitern,
        # oder innerhalb der Methode, wenn die Klasse auch ohne PyDub instanziierbar sein soll.
        try:
            from pydub import AudioSegment, exceptions as pydub_exceptions
            self._AudioSegment = AudioSegment
            self._pydub_exceptions = pydub_exceptions
        except ImportError:
            logger.error("PyDub nicht verfügbar - installieren mit: pip install pydub")
            # Erwägen, hier einen Fehler auszulösen, wenn PyDub eine harte Abhängigkeit ist.
            raise

    def load_audio(
        self,
        file_path: Union[str, Path],
        sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> np.ndarray:
        """Lädt Audio mit PyDub, macht es robuster gegenüber Fehlern."""

        # Stelle sicher, dass PyDub verfügbar ist (falls nicht im Konstruktor geprüft)
        if not hasattr(self, '_AudioSegment'):
             raise RuntimeError("PyDub wurde nicht korrekt initialisiert oder ist nicht installiert.")

        file_path_obj = Path(file_path) # Konvertiere zu Path für einfache Handhabung
        if not file_path_obj.is_file():
            logger.error(f"Audiodatei nicht gefunden: {file_path_obj}")
            raise FileNotFoundError(f"Audiodatei nicht gefunden: {file_path_obj}")

        file_path_str = str(file_path_obj)
        logger.debug(f"🎵 Lade Audio mit PyDub: {file_path_str}")

        try:
            audio_segment = self._AudioSegment.from_file(file_path_str)
        except self._pydub_exceptions.CouldntDecodeError as e:
            logger.error(
                f"PyDub konnte die Audiodatei nicht dekodieren: {file_path_str}. "
                f"Möglicherweise fehlt ffmpeg/avconv oder das Format wird nicht unterstützt. Fehler: {e}"
            )
            raise RuntimeError(f"PyDub Dekodierungsfehler für {file_path_str}") from e
        except Exception as e: # Andere mögliche Fehler beim Laden (z.B. Berechtigungen)
            logger.error(f"Unerwarteter Fehler beim Laden von '{file_path_str}' mit PyDub: {e}", exc_info=True)
            raise RuntimeError(f"Allgemeiner Fehler beim Laden von '{file_path_str}' mit PyDub") from e

        # Validierung und Anwendung von Offset und Duration
        total_duration_ms = len(audio_segment)
        start_ms = int(offset * 1000)
        
        if offset < 0:
            logger.warning(f"Negativer Offset ({offset}s) wird als 0s behandelt.")
            start_ms = 0
        
        if duration is not None:
            if duration <= 0:
                logger.warning(f"Nicht-positive Dauer ({duration}s) wird ignoriert; Audio wird ab Offset bis zum Ende geladen.")
                end_ms = total_duration_ms
            else:
                end_ms = start_ms + int(duration * 1000)
        else:
            end_ms = total_duration_ms # Gesamte Länge ab start_ms

        # Stelle sicher, dass Schnittgrenzen valide sind
        start_ms = max(0, start_ms)
        end_ms = min(total_duration_ms, end_ms) # Nicht über das Ende hinaus

        if start_ms >= end_ms or start_ms >= total_duration_ms:
            logger.warning(
                f"Schnittbereich (Start: {start_ms}ms, Ende: {end_ms}ms bei Gesamtlänge {total_duration_ms}ms) "
                f"für Offset {offset}s und Dauer {duration}s ist leer oder ungültig. "
                "Gebe leeres Audio zurück."
            )
            return np.array([], dtype=np.float32)
        
        # Schneiden nur wenn nötig (start_ms ist nicht 0 ODER end_ms ist nicht die Gesamtlänge)
        if start_ms > 0 or end_ms < total_duration_ms:
            logger.debug(f"Schneide Audio von {start_ms}ms bis {end_ms}ms.")
            audio_segment = audio_segment[start_ms:end_ms]

        if len(audio_segment) == 0: # Sollte durch obige Prüfung abgedeckt sein, aber als Sicherheitsnetz
            logger.warning("Audio-Segment ist nach dem Schneiden leer.")
            return np.array([], dtype=np.float32)

        original_sr = audio_segment.frame_rate
        samples = np.array(audio_segment.get_array_of_samples())

        if samples.size == 0:
            logger.warning("Konvertierung zu NumPy ergab ein leeres Array trotz nicht-leeren Segments (unerwartet).")
            return np.array([], dtype=np.float32)
        
        # Normalisiere von Integer zu Float32 [-1.0, 1.0]
        # sample_width ist in Bytes (z.B. 2 für 16-bit, 4 für 32-bit)
        if audio_segment.sample_width == 2:  # 16-bit signed
            norm_factor = 32768.0  # 2^15
        elif audio_segment.sample_width == 4:  # 32-bit signed
            norm_factor = 2147483648.0  # 2^31
        elif audio_segment.sample_width == 1: # 8-bit signed (pydub konvertiert intern zu signed)
            norm_factor = 128.0 # 2^7
        else:
            bits = audio_segment.sample_width * 8
            if bits > 0:
                norm_factor = float(1 << (bits - 1))
                logger.warning(
                    f"Unerwartete Sample-Breite: {audio_segment.sample_width} bytes ({bits}-bit). "
                    f"Verwende generischen Normalisierungsfaktor {norm_factor}. Dies könnte ungenau sein."
                )
            else:
                logger.error(f"Ungültige Sample-Breite ({audio_segment.sample_width} bytes) für Normalisierung. Normalisierung übersprungen.")
                norm_factor = 1.0 # Verhindert Division durch Null, aber Daten sind wahrscheinlich falsch skaliert

        audio_float = samples.astype(np.float32) / norm_factor
        audio_float = np.clip(audio_float, -1.0, 1.0) # Wertebereich sicherstellen

        # Stereo zu Mono / Kanalbehandlung
        num_channels = audio_segment.channels
        if num_channels > 1:
            if audio_float.size % num_channels == 0:
                audio_float = audio_float.reshape((-1, num_channels))
                if mono:
                    audio_float = np.mean(audio_float, axis=1)
            else:
                logger.error(
                    f"Audio-Array-Länge ({audio_float.size}) nicht ganzzahlig durch Kanalanzahl ({num_channels}) teilbar. "
                    "Kanalverarbeitung könnte fehlerhaft sein. Gebe Audio unverändert zurück."
                )
                # Hier könnte man entscheiden, ob man einen Fehler wirft oder das (potenziell fehlerhafte) Audio zurückgibt.
                # Für Robustheit ggü. Absturz: Rückgabe des bisherigen Zustands.
        elif num_channels == 1 and not mono:
            # Audio ist bereits mono, aber es wurde explizit nicht mono angefordert.
            # Es bleibt mono, da aus Mono nicht sinnvoll Stereo erzeugt werden kann ohne Duplizierung.
            logger.debug("Audio ist bereits Mono. mono=False führt zu Mono-Ausgabe.")
        # Wenn num_channels == 1 und mono == True, ist nichts zu tun.

        # Resampling falls nötig mit librosa
        if original_sr != sr:
            if audio_float.size == 0:
                logger.debug("Leeres Audio, kein Resampling erforderlich.")
            else:
                try:
                    import librosa
                except ImportError:
                    logger.error(
                        "Librosa nicht verfügbar, aber für Resampling von "
                        f"{original_sr}Hz zu {sr}Hz erforderlich. "
                        "Installieren mit: pip install librosa"
                    )
                    raise RuntimeError(
                        "Resampling fehlgeschlagen: Librosa nicht installiert."
                    ) from ImportError # Wichtig für klare Fehlermeldung

                logger.debug(f"Resample von {original_sr}Hz zu {sr}Hz mit Librosa.")
                
                # Librosa erwartet für Multichannel (channels, samples), also (num_channels, num_samples_per_channel)
                # Aktuell ist audio_float entweder 1D (mono) oder (num_samples_per_channel, num_channels)
                y_for_resample = audio_float
                needs_transpose_back = False

                if audio_float.ndim == 2 and audio_float.shape[1] == num_channels and num_channels > 1:
                    y_for_resample = audio_float.T # Zu (num_channels, num_samples_per_channel)
                    needs_transpose_back = True
                
                # librosa.resample kann ValueError werfen, z.B. wenn Daten nicht endlich sind
                try:
                    resampled_audio = librosa.resample(y_for_resample, orig_sr=original_sr, target_sr=sr)
                except ValueError as ve:
                    logger.error(f"Librosa Resampling Fehler: {ve}. Möglicherweise nicht-endliche Werte im Audio.", exc_info=True)
                    raise RuntimeError(f"Librosa Resampling Fehler für {file_path_str}") from ve


                if needs_transpose_back and resampled_audio.ndim == 2:
                    # Stelle sicher, dass die Form nach dem Resampling noch Sinn ergibt
                    if resampled_audio.shape[0] == num_channels:
                         audio_float = resampled_audio.T # Zurück zu (num_samples_per_channel, num_channels)
                    else: # Unerwartete Form nach Resampling, z.B. wenn es zu Mono wurde
                        logger.warning(f"Unerwartete Form nach Librosa Resampling ({resampled_audio.shape}), nicht zurücktransponiert.")
                        audio_float = resampled_audio

                else: # War Mono oder wurde zu Mono
                    audio_float = resampled_audio
        
        logger.debug(f"✅ PyDub geladen und verarbeitet: Shape {audio_float.shape}, SR {sr}Hz, Mono={mono}")
        return audio_float.astype(np.float32) # Stelle sicher, dass es Float32 ist

    def get_strategy_name(self) -> str:
        return "pydub"


class AudioValidator:
    """Validiert Audio-Dateien gegen verschiedene Constraints"""
    
    def __init__(
        self, 
        max_duration: float = 7200,  # 2 hours
        min_sample_rate: int = 8000,
        max_file_size: int = 500 * 1024 * 1024,  # 500MB
        supported_formats: Optional[list] = None
    ):
        self.max_duration = max_duration
        self.min_sample_rate = min_sample_rate
        self.max_file_size = max_file_size
        self.supported_formats = supported_formats or [f.value for f in AudioFormat]
    
    async def validate_file_async(
        self, 
        file_path: Path, 
        original_filename: str
    ) -> Tuple[bool, str, Optional[AudioMetadata]]:
        """
        Asynchrone Audio-Validierung.
        
        Args:
            file_path: Pfad zur Audio-Datei
            original_filename: Original-Dateiname
            
        Returns:
            Tuple[bool, str, AudioMetadata]: (is_valid, message, metadata)
        """
        try:
            # Basic file checks
            if not file_path.exists():
                return False, "File not found", None
            
            file_size = file_path.stat().st_size
            
            # File size check
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size/(1024*1024):.1f}MB (max: {self.max_file_size/(1024*1024):.1f}MB)", None
            
            if file_size < 1000:  # 1KB minimum
                return False, "File too small or corrupted", None
            
            # Format check
            extension = file_path.suffix.lower().lstrip('.')
            if extension not in self.supported_formats:
                return False, f"Unsupported format: {extension}. Supported: {self.supported_formats}", None
            
            # Audio content validation (async)
            def _validate_audio_content():
                return self._validate_audio_properties(file_path, file_size, extension)
            
            is_valid, message, metadata = await asyncio.to_thread(_validate_audio_content)
            return is_valid, message, metadata
            
        except Exception as e:
            logger.error(f"❌ Validation error for {original_filename}: {e}")
            return False, f"Validation error: {e}", None
    
    def _validate_audio_properties(
        self, 
        file_path: Path, 
        file_size: int, 
        format_ext: str
    ) -> Tuple[bool, str, Optional[AudioMetadata]]:
        """Validiert Audio-Eigenschaften (synchron)"""
        try:
            # Versuche Audio-Metadaten zu extrahieren
            metadata = self._extract_metadata(file_path, file_size, format_ext)
            
            # Duration check
            if metadata.duration_seconds > self.max_duration:
                return False, f"Audio too long: {metadata.duration_seconds/3600:.1f}h (max: {self.max_duration/3600:.1f}h)", metadata
            
            if metadata.duration_seconds < 1.0:
                return False, "Audio too short (min: 1 second)", metadata
            
            # Sample rate check
            if metadata.sample_rate < self.min_sample_rate:
                return False, f"Sample rate too low: {metadata.sample_rate}Hz (min: {self.min_sample_rate}Hz)", metadata
            
            logger.info(f"✅ Audio validated: {metadata.duration_seconds:.2f}s, {metadata.sample_rate}Hz, {format_ext.upper()}")
            return True, "OK", metadata
            
        except Exception as e:
            return False, f"Audio property validation failed: {e}", None
    
    def _extract_metadata(self, file_path: Path, file_size: int, format_ext: str) -> AudioMetadata:
        """Extrahiert Audio-Metadaten"""
        try:
            # Versuche SoundFile für Metadaten
            with sf.SoundFile(str(file_path)) as f:
                return AudioMetadata(
                    duration_seconds=len(f) / f.samplerate,
                    sample_rate=f.samplerate,
                    channels=f.channels,
                    file_size_bytes=file_size,
                    format=format_ext,
                    bit_depth=getattr(f, 'subtype_info', {}).get('bits_per_sample')
                )
        except Exception:
            try:
                # Fallback: Librosa
                y, sr = librosa.load(str(file_path), sr=None, duration=1.0)  # Nur 1s für Metadaten
                
                # Schätze Gesamtdauer
                estimated_duration = file_size / (sr * 2 * 1.5)  # Grobe Schätzung
                
                return AudioMetadata(
                    duration_seconds=estimated_duration,
                    sample_rate=sr,
                    channels=1 if len(y.shape) == 1 else y.shape[0],
                    file_size_bytes=file_size,
                    format=format_ext,
                    validation_message="Estimated metadata (librosa fallback)"
                )
            except Exception as e:
                raise AudioValidationError(f"Could not extract audio metadata: {e}")


class AudioLoader:
    """
    Zentraler Audio-Loader mit Multiple-Strategy-Pattern.
    
    Wie ein schlauer DJ, der verschiedene Plattenspieler hat:
    - Erst versucht er den guten Technics (SoundFile)
    - Falls der spinnt, nimmt er den Backup-Player (Librosa)  
    - Als Notlösung den alten Kassettenrekorder (PyDub)
    """
    
    def __init__(self):
        # Strategien in Prioritäts-Reihenfolge
        self.strategies = [
            SoundFileStrategy(),
            LibrosaStrategy(),
            PyDubStrategy(),
        ]
        
        # Format-zu-Strategie-Mapping für Optimierung
        self.format_preferences = {
            AudioFormat.WAV: [SoundFileStrategy(), LibrosaStrategy()],
            AudioFormat.FLAC: [SoundFileStrategy(), LibrosaStrategy()],
            AudioFormat.OGG: [SoundFileStrategy(), LibrosaStrategy()],
            AudioFormat.MP3: [LibrosaStrategy(), PyDubStrategy()],
            AudioFormat.MP4: [PyDubStrategy(), LibrosaStrategy()],
            AudioFormat.M4A: [PyDubStrategy(), LibrosaStrategy()],
        }
    
    async def load_audio_async(
        self,
        file_path: Union[str, Path],
        sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
        format_hint: Optional[str] = None
    ) -> np.ndarray:
        """
        Lädt Audio-Datei asynchron mit automatischer Strategie-Auswahl.
        
        Args:
            file_path: Pfad zur Audio-Datei
            sr: Ziel-Sample-Rate
            mono: Zu Mono konvertieren
            duration: Maximale Dauer zu laden
            offset: Start-Offset in Sekunden
            format_hint: Format-Hint für Strategie-Optimierung
            
        Returns:
            np.ndarray: Geladene Audio-Daten
            
        Raises:
            AudioValidationError: Wenn alle Strategien fehlschlagen
        """
        file_path = Path(file_path)
        
        logger.info(f"🎵 Loading audio: {file_path.name} (sr={sr}, mono={mono})")
        start_time = time.time()
        
        # Bestimme optimale Strategien basierend auf Format
        strategies_to_try = self._get_strategies_for_format(format_hint or file_path.suffix.lower().lstrip('.'))
        
        def _load_audio():
            return self._load_audio_sync(file_path, sr, mono, duration, offset, strategies_to_try)
        
        try:
            audio_data = await asyncio.to_thread(_load_audio)
            load_time = time.time() - start_time
            
            logger.info(f"✅ Audio loaded: {audio_data.shape} @ {sr}Hz in {load_time:.2f}s")
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio loading failed for {file_path.name}: {e}")
            raise AudioValidationError(f"Could not load audio file: {e}")
    
    def _load_audio_sync(
        self,
        file_path: Path,
        sr: int,
        mono: bool,
        duration: Optional[float],
        offset: float,
        strategies: list
    ) -> np.ndarray:
        """Synchrone Audio-Loading-Logik"""
        
        last_error = None
        
        for strategy in strategies:
            try:
                logger.debug(f"🎯 Trying strategy: {strategy.get_strategy_name()}")
                
                audio = strategy.load_audio(
                    file_path=file_path,
                    sr=sr,
                    mono=mono,
                    duration=duration,
                    offset=offset
                )
                
                # Validiere Ergebnis
                if audio is None or len(audio) == 0:
                    raise ValueError("No audio data loaded")
                
                logger.debug(f"✅ Success with {strategy.get_strategy_name()}")
                return audio
                
            except Exception as e:
                logger.debug(f"⚠️ Strategy {strategy.get_strategy_name()} failed: {e}")
                last_error = e
                continue
        
        # Alle Strategien fehlgeschlagen
        raise AudioValidationError(f"All loading strategies failed. Last error: {last_error}")
    
    def _get_strategies_for_format(self, format_ext: str) -> list:
        """Gibt optimale Strategien für ein Format zurück"""
        try:
            audio_format = AudioFormat(format_ext.lower())
            return self.format_preferences.get(audio_format, self.strategies)
        except ValueError:
            # Unbekanntes Format - verwende alle Strategien
            return self.strategies


class AudioService:
    """
    Haupt-Audio-Service-Klasse.
    
    Der Chefdirigent des Audio-Orchesters - koordiniert Validation,
    Loading und alle anderen Audio-Operationen.
    """
    
    def __init__(
        self,
        max_duration: float = 7200,
        min_sample_rate: int = 8000,
        max_file_size: int = 500 * 1024 * 1024,
        supported_formats: Optional[list] = None
    ):
        self.validator = AudioValidator(
            max_duration=max_duration,
            min_sample_rate=min_sample_rate,
            max_file_size=max_file_size,
            supported_formats=supported_formats
        )
        self.loader = AudioLoader()
    
    async def validate_and_load_audio(
        self,
        file_path: Union[str, Path],
        original_filename: str,
        sr: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, AudioMetadata]:
        """
        Validiert und lädt Audio-Datei in einem Schritt.
        
        Args:
            file_path: Pfad zur Audio-Datei
            original_filename: Original-Dateiname
            sr: Ziel-Sample-Rate
            mono: Zu Mono konvertieren
            duration: Maximale Dauer
            offset: Start-Offset
            
        Returns:
            Tuple[np.ndarray, AudioMetadata]: (audio_data, metadata)
            
        Raises:
            AudioValidationError: Bei Validierungs- oder Loading-Fehlern
        """
        file_path = Path(file_path)
        
        # 1. Validierung
        is_valid, message, metadata = await self.validator.validate_file_async(file_path, original_filename)
        
        if not is_valid:
            raise AudioValidationError(f"Audio validation failed: {message}")
        
        # 2. Loading
        format_hint = file_path.suffix.lower().lstrip('.')
        audio_data = await self.loader.load_audio_async(
            file_path=file_path,
            sr=sr,
            mono=mono,
            duration=duration,
            offset=offset,
            format_hint=format_hint
        )
        
        return audio_data, metadata
    
    async def validate_audio_file(
        self,
        file_path: Union[str, Path],
        original_filename: str
    ) -> Tuple[bool, str, Optional[AudioMetadata]]:
        """
        Validiert Audio-Datei ohne Loading.
        
        Args:
            file_path: Pfad zur Datei
            original_filename: Original-Dateiname
            
        Returns:
            Tuple[bool, str, AudioMetadata]: (is_valid, message, metadata)
        """
        return await self.validator.validate_file_async(Path(file_path), original_filename)
    
    async def load_audio_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> np.ndarray:
        """
        Lädt Audio-Datei (ohne Validierung).
        
        Args:
            file_path: Pfad zur Datei
            **kwargs: Parameter für Audio-Loading
            
        Returns:
            np.ndarray: Audio-Daten
        """
        return await self.loader.load_audio_async(file_path, **kwargs)
    
    def is_supported_format(self, file_extension: str) -> bool:
        """Prüft ob Format unterstützt wird"""
        ext = file_extension.lower().lstrip('.')
        return ext in self.validator.supported_formats
    
    def get_supported_formats(self) -> list:
        """Gibt unterstützte Formate zurück"""
        return self.validator.supported_formats.copy()


# Default Audio Service Instance
default_audio_service = AudioService()


# Convenience Functions
async def validate_and_load_audio(file_path, original_filename: str, **kwargs):
    """Convenience function für Audio-Validierung und Loading"""
    return await default_audio_service.validate_and_load_audio(file_path, original_filename, **kwargs)


async def validate_audio_file(file_path, original_filename: str):
    """Convenience function für Audio-Validierung"""
    return await default_audio_service.validate_audio_file(file_path, original_filename)


def is_supported_format(file_extension: str) -> bool:
    """Convenience function für Format-Check"""
    return default_audio_service.is_supported_format(file_extension)