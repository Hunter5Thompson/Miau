# services/alignment_service.py

import asyncio
import logging
import time
import gc
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import whisperx  # type: ignore

logger = logging.getLogger(__name__)


# Versuche, die echten Klassen zu importieren; bei Fehlschlag Dummy-Klassen definieren
try:
    from services.transcription_service import TranscriptionResult
    from config.hardware_detection import HardwareCapabilities
except ImportError as e:
    logger.error(f"ImportError beim Laden von Abhängigkeiten: {e}. Verwende Dummy-Klassen.")
    @dataclass
    class TranscriptionResult: # type: ignore
        language: Optional[str] = None
        segments: List[Any] = field(default_factory=list)

    @dataclass
    class HardwareCapabilities: # type: ignore
        device: str = "cpu"
        gpu_vram_gb: Optional[float] = None
        gpu_name: Optional[str] = None # Hinzugefügt basierend auf ml_pipeline Nutzung
        gpu_memory_gb: Optional[float] = None # Hinzugefügt basierend auf ml_pipeline Nutzung


class LocalModelResolver:
    def __init__(self, cache_base_dir: str):
        self.cache_base = Path(cache_base_dir).resolve()
        self.local_models = {
            "de": {
                "search_patterns": [
                    "models--jonatasgrosman--wav2vec2-large-xlsr-53-german",
                    "hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-german",
                    "jonatasgrosman/wav2vec2-large-xlsr-53-german",
                    "wav2vec2-large-xlsr-53-german",
                ]
            }
        }
        logger.info(f"LocalModelResolver init – Cache-Pfad: {self.cache_base}")
        self._debug_cache_structure()

    def _debug_cache_structure(self):
        exists = self.cache_base.exists()
        logger.info(f"Cache-Basis existiert: {exists} ({self.cache_base})")
        if exists and self.cache_base.is_dir():
            try:
                contents = [f.name for f in self.cache_base.iterdir()][:5]
                logger.info(f"Cache-Inhalt (erste 5): {contents}")
            except Exception as e:
                logger.warning(f"Kann Verzeichnis nicht auflisten: {self.cache_base} – {e}")

    def find_local_alignment_model(self, language_code: str) -> Optional[Path]:
        logger.info(f"Suche nach lokalem Modell für Sprache: '{language_code}'")
        cfg = self.local_models.get(language_code)
        if cfg is None:
            logger.warning(f"Sprache '{language_code}' nicht konfiguriert für lokale Suche.")
            return None

        hf_cache_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")).resolve()
        transformers_cache = Path(os.getenv("TRANSFORMERS_CACHE", hf_cache_home)).resolve()

        bases = [
            self.cache_base,
            self.cache_base / "hub",
            transformers_cache,
            transformers_cache / "hub",
            hf_cache_home / "hub",
            Path("/app/models").resolve(),
            Path("/models").resolve(),
            Path("/app/.cache").resolve(),
        ]
        unique_bases = sorted({p for p in bases if p.is_dir()}, key=lambda p: str(p))
        logger.info(f"Zu prüfende Basisverzeichnisse: {unique_bases}")

        paths_to_try = []
        for base_dir in unique_bases:
            for pattern in cfg["search_patterns"]:
                paths_to_try.append(base_dir / pattern)

        unique_paths = sorted(
            {p.resolve(strict=False) for p in paths_to_try},
            key=lambda p: str(p)
        )
        logger.info(f"Es werden {len(unique_paths)} eindeutige Pfade für '{language_code}' geprüft.")

        for i, path_obj in enumerate(unique_paths):
            logger.info(f"  Prüfe Pfad {i+1}/{len(unique_paths)}: {path_obj}")
            if self._validate_wav2vec2_model(path_obj):
                logger.info(f"  ✅ Validiertes Modell gefunden: {path_obj}")
                return path_obj
            elif path_obj.is_dir():
                logger.info(f"  ⚠️ Pfad {path_obj} ist ein Verzeichnis, wurde aber nicht validiert (siehe 🕵️ Logs von _validate).")
            # else: logger.info(f"  ℹ️ Pfad {path_obj} existiert nicht oder ist kein Verzeichnis.") # Redundant mit _validate logs

        logger.warning(f"Kein gültiges lokales Modell für '{language_code}' gefunden nach Prüfung von {len(unique_paths)} Pfaden.")
        return None

    def _validate_wav2vec2_model(self, path_to_validate: Path) -> bool:
        logger.info(f"🕵️ Validierung für: {path_to_validate}")
        if not path_to_validate.is_dir():
            logger.info(f"🕵️ Pfad ist kein Verzeichnis oder existiert nicht.")
            return False
        try:
            items = {item.name.lower(): item for item in path_to_validate.iterdir()}
            files_lower = {name: item for name, item in items.items() if item.is_file()}
            dirs_lower = {name: item for name, item in items.items() if item.is_dir()}
            logger.info(
                f"🕵️ Inhalt (iterdir): Dateien (max 10): {list(files_lower.keys())[:10]}, "
                f"Ordner (max 5): {list(dirs_lower.keys())[:5]}"
            )
        except Exception as e:
            logger.warning(f"🕵️ Kann {path_to_validate} nicht iterieren: {e}"); return False

        required_direct = {"pytorch_model.bin", "config.json"}
        found_direct = {f for f in required_direct if f in files_lower}
        logger.info(f"🕵️ S1: Direkte Dateien. Benötigt: {required_direct}. Gefunden (iterdir): {found_direct}")
        if found_direct == required_direct:
            logger.info(f"✅ S1 ERFOLG: Direkte Struktur OK: {path_to_validate}"); return True
        
        snapshots_dir_obj = dirs_lower.get("snapshots")
        if snapshots_dir_obj:
            logger.info(f"🕵️ S2: 'snapshots' Ordner gefunden: {snapshots_dir_obj}. Prüfe Unterordner...")
            try:
                for subdir in snapshots_dir_obj.iterdir():
                    if not subdir.is_dir(): continue
                    logger.info(f"🕵️ S2: Prüfe Snapshot-Unterordner: {subdir}")
                    try: sub_files_lower = {f.name.lower() for f in subdir.iterdir() if f.is_file()}
                    except Exception as e_subdir: logger.warning(f"🕵️ S2: Fehler beim Iterieren von {subdir}: {e_subdir}"); continue
                    found_snapshot_files = {f for f in required_direct if f in sub_files_lower}
                    logger.info(f"🕵️ S2: Dateien in '{subdir.name}'. Benötigt: {required_direct}. Gefunden: {found_snapshot_files}")
                    if found_snapshot_files == required_direct:
                        logger.info(f"✅ S2 ERFOLG: Snapshot-Struktur OK in {subdir}"); return True
                    if {"model.safetensors", "config.json"}.issubset(sub_files_lower):
                        logger.info(f"✅ S2 ERFOLG (safetensors): Snapshot-Struktur OK in {subdir}"); return True
            except Exception as e_snap_scan: logger.warning(f"🕵️ S2: Fehler beim Scannen der Snapshots in {snapshots_dir_obj}: {e_snap_scan}")
        else: logger.info(f"🕵️ S2: Kein 'snapshots' Ordner in {path_to_validate} gefunden.")

        config_at_root = "config.json" in files_lower
        if config_at_root and ("refs" in dirs_lower or "blobs" in dirs_lower):
            logger.info(f"✅ S3/4 ERFOLG (Heuristik): 'config.json' und ('refs' oder 'blobs') OK in {path_to_validate}")
            return True
        logger.info(f"🕵️ S3/4: Heuristik. config.json @ root: {config_at_root}, refs exist: {'refs' in dirs_lower}, blobs exist: {'blobs' in dirs_lower}")
        logger.info(f"❌ Alle Validierungsstrategien fehlgeschlagen für {path_to_validate}.")
        return False

class AlignmentQuality(Enum):
    BASIC = "basic"; PRECISE = "precise"; PHONEME = "phoneme"

@dataclass
class AlignmentConfig:
    quality: AlignmentQuality = AlignmentQuality.PRECISE
    return_char_alignments: bool = False
    language_code: str = "de"
    interpolate_method: str = "nearest"
    extend_duration: float = 0.0
    use_token_timestamps: bool = True
    force_phoneme_alignment: bool = False

    @classmethod
    def get_basic_config(cls, language: str = "de") -> "AlignmentConfig":
        return cls(quality=AlignmentQuality.BASIC, language_code=language, return_char_alignments=False)
    @classmethod
    def get_precise_config(cls, language: str = "de") -> "AlignmentConfig":
        return cls(quality=AlignmentQuality.PRECISE, return_char_alignments=True, language_code=language, interpolate_method="linear")
    @classmethod
    def get_phoneme_config(cls, language: str = "de") -> "AlignmentConfig":
        return cls(quality=AlignmentQuality.PHONEME, return_char_alignments=True, language_code=language, interpolate_method="linear", force_phoneme_alignment=True)

@dataclass
class WordAlignment:
    word: str; start: float; end: float
    confidence: Optional[float] = None
    char_alignments: Optional[List[Dict[str, Any]]] = None

@dataclass
class AlignedSegment:
    start: float; end: float; text: str
    words: List[WordAlignment]
    confidence: Optional[float] = None

@dataclass
class AlignmentResult:
    segments: List[AlignedSegment]; language: str; model_name: Optional[str]
    config: AlignmentConfig; processing_time: float
    model_load_time: Optional[float] = None; alignment_time: Optional[float] = None
    total_words: int = 0; average_confidence: Optional[float] = None
    def __post_init__(self):
        self.total_words = sum(len(s.words) for s in self.segments)
        c = [w.confidence for s in self.segments for w in s.words if w.confidence is not None]
        if c: self.average_confidence = sum(c) / len(c)

class AlignmentModelCache:
    def __init__(self):
        self.models: Dict[str, Tuple[Any, Any]] = {}
        self.load_times: Dict[str, float] = {}
        logger.info("AlignmentModelCache initialisiert.")
    def get_model(self, language: str, device: str) -> Optional[Tuple[Any, Any]]:
        key = f"{language}_{device}"; model_tuple = self.models.get(key)
        logger.info(f"Cache-Abfrage für '{key}': {'Treffer' if model_tuple else 'Kein Treffer'}")
        return model_tuple
    def cache_model(self, language: str, device: str, model: Any, meta: Any, load_time: float):
        key = f"{language}_{device}"; self.models[key] = (model, meta); self.load_times[key] = load_time
        logger.info(f"Modell '{key}' für {load_time:.2f}s im Cache gespeichert.")
    def clear_cache(self):
        keys = list(self.models.keys()); logger.info(f"Cache wird geleert: {keys}")
        self.models.clear(); self.load_times.clear(); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info("Cache erfolgreich geleert.")
    def get_cache_info(self) -> Dict[str, Any]:
        return {"cached_models": list(self.models.keys()), "load_times": self.load_times.copy(), "total_models": len(self.models)}

class AlignmentEngine:
    def __init__(self, hw_capabilities: HardwareCapabilities, model_download_cache_dir: str):
        self.hw = hw_capabilities
        self.whisperx_download_dir = Path(model_download_cache_dir).resolve()
        self.resolver = LocalModelResolver(model_download_cache_dir)
        self.mem_cache = AlignmentModelCache()
        logger.info(f"AlignmentEngine init: WhisperX Download-Dir: {self.whisperx_download_dir}, Device: {self.hw.device}")

    async def _load_alignment_model(self, cfg: AlignmentConfig) -> Tuple[Any, Any]:
        cached_model, cached_meta = self.mem_cache.get_model(cfg.language_code, self.hw.device) or (None, None)
        if cached_model: return cached_model, cached_meta

        load_operation_start_time = time.time()
        loaded_model, loaded_meta = None, None
        model_path_used_for_loading = "N/A"

        local_model_path = self.resolver.find_local_alignment_model(cfg.language_code)
        if local_model_path:
            model_path_used_for_loading = str(local_model_path)
            logger.info(f"Versuche, lokal gefundenes Modell zu laden: {model_path_used_for_loading}")
            try:
                def _load_local_task():
                    return whisperx.load_align_model(None, self.hw.device, model_path_used_for_loading, str(self.whisperx_download_dir))
                loaded_model, loaded_meta = await asyncio.to_thread(_load_local_task)
                loaded_meta = loaded_meta or {}; loaded_meta.update({"model_path": model_path_used_for_loading, "is_local_model": True})
                logger.info(f"✅ Lokales Modell erfolgreich geladen: {model_path_used_for_loading}")
            except Exception as e_local_load:
                logger.warning(f"⚠️ Fehler beim Laden des lokalen Modells '{model_path_used_for_loading}': {e_local_load}. Versuche Hub-Fallback.")
                loaded_model = None

        if loaded_model is None:
            logger.info(f"Kein lokales Modell geladen. Versuche Download vom Hub für Sprache '{cfg.language_code}'.")
            allow_online_dl = os.getenv("ALLOW_ONLINE_MODEL_DOWNLOAD", "false").lower() == "true"
            is_hf_offline_mode = os.getenv("HF_HUB_OFFLINE", "0") == "1"
            if is_hf_offline_mode and not allow_online_dl:
                error_msg = f"Offline-Modus (HF_HUB_OFFLINE=1) aktiv und Online-Download (ALLOW_ONLINE_MODEL_DOWNLOAD=false) nicht erlaubt für Sprache '{cfg.language_code}'."
                logger.error(f"❌ {error_msg}"); raise RuntimeError(error_msg)
            if is_hf_offline_mode and allow_online_dl:
                logger.warning(f"⚠️ Offline-Modus (HF_HUB_OFFLINE=1) ist aktiv, aber Online-Download ist ERLAUBT (ALLOW_ONLINE_MODEL_DOWNLOAD=true). Versuche Download für '{cfg.language_code}'.")
            try:
                def _load_hub_task():
                    return whisperx.load_align_model(cfg.language_code, self.hw.device, model_dir=str(self.whisperx_download_dir))
                loaded_model, loaded_meta = await asyncio.to_thread(_load_hub_task)
                loaded_meta = loaded_meta or {}
                model_path_used_for_loading = loaded_meta.get("model_path", f"Hub-Modell für {cfg.language_code}")
                loaded_meta.update({"model_path": model_path_used_for_loading, "is_local_model": False})
                logger.info(f"✅ Hub-Modell für '{cfg.language_code}' erfolgreich geladen/gefunden.")
            except Exception as e_hub_load:
                logger.error(f"❌ Fehler beim Laden des Hub-Modells für '{cfg.language_code}': {e_hub_load}", exc_info=True)
                raise RuntimeError(f"Fehler beim Laden des Hub-Modells für {cfg.language_code}: {e_hub_load}") from e_hub_load

        model_load_duration = time.time() - load_operation_start_time
        self.mem_cache.cache_model(cfg.language_code, self.hw.device, loaded_model, loaded_meta, model_load_duration)
        return loaded_model, loaded_meta

    async def align_transcription(
        self, segments: List[Dict[str, Any]], audio: np.ndarray, sr: int,
        cfg: AlignmentConfig, progress_callback: Optional[callable] = None
    ) -> AlignmentResult:
        total_start_time = time.time()
        logger.info(f"Alignment gestartet: {len(segments)} Segmente, Sprache: {cfg.language_code}, Device: {self.hw.device}")
        if sr != 16000: logger.warning(f"Samplerate {sr}Hz. Alignment-Modelle erwarten 16000Hz.")
        if progress_callback: progress_callback("alignment.loading_model", 0.1)

        model_load_start_time = time.time()
        loaded_model, model_metadata = await self._load_alignment_model(cfg)
        model_load_total_time = time.time() - model_load_start_time
        logger.info(f"Laden Alignment-Modell: {model_load_total_time:.2f}s.")
        if progress_callback: progress_callback("alignment.aligning_segments", 0.3)

        alignment_op_start_time = time.time()
        raw_whisperx_result: Dict[str, List[Dict[str, Any]]] = {"segments": []}
        if not segments: logger.warning("Keine Segmente zum Alignen übergeben.")
        else:
            prepared_segments = [{"text": s.get("text", ""), "start": s.get("start"), "end": s.get("end")} for s in segments]
            def _run_align_in_thread():
                logger.info(f"Starte whisperx.align mit {len(prepared_segments)} Segmenten.")
                return whisperx.align(prepared_segments, loaded_model, model_metadata, audio, self.hw.device,
                                      language_code=cfg.language_code, return_char_alignments=cfg.return_char_alignments,
                                      interpolate_method=cfg.interpolate_method, print_progress=False)
            try:
                raw_whisperx_result = await asyncio.to_thread(_run_align_in_thread)
            except Exception as e_align_op:
                logger.error(f"Fehler während whisperx.align: {e_align_op}", exc_info=True)
                raise RuntimeError(f"whisperx.align fehlgeschlagen: {e_align_op}") from e_align_op
        alignment_op_total_time = time.time() - alignment_op_start_time
        logger.info(f"Alignment-Operation (whisperx.align): {alignment_op_total_time:.2f}s.")
        if progress_callback: progress_callback("alignment.processing_results", 0.8)
        
        segments_from_engine = raw_whisperx_result.get("segments", raw_whisperx_result.get("word_segments", []))
        processed_segments = self._process_whisperx_output(segments_from_engine, cfg)
        total_processing_time = time.time() - total_start_time
        
        final_result = AlignmentResult(processed_segments, cfg.language_code, model_metadata.get("model_path", "Unbekannt"),
                                     cfg, total_processing_time, model_load_total_time, alignment_op_total_time)
        if progress_callback: progress_callback("alignment.completed", 1.0)
        logger.info(f"✅ Alignment abgeschlossen: {final_result.total_words} Wörter in {total_processing_time:.2f}s")
        return final_result

    def _process_whisperx_output(self, whisperx_segments: List[Dict[str, Any]], config: AlignmentConfig) -> List[AlignedSegment]:
        output_aligned_segments = []
        if not whisperx_segments: logger.warning("Keine Rohsegmente von whisperx zum Verarbeiten."); return output_aligned_segments
        for i, raw_segment_dict in enumerate(whisperx_segments):
            if not isinstance(raw_segment_dict, dict): logger.warning(f"Segment {i} ist kein Dict."); continue
            word_objects_for_segment: List[WordAlignment] = []
            raw_words_list = raw_segment_dict.get("words", [])
            if not isinstance(raw_words_list, list): raw_words_list = []
            for j, raw_word_dict in enumerate(raw_words_list):
                if not isinstance(raw_word_dict, dict): logger.warning(f"Wort {j} in Seg {i} kein Dict."); continue
                word_text = str(raw_word_dict.get("word", "")).strip()
                start_time, end_time = raw_word_dict.get("start"), raw_word_dict.get("end")
                confidence_score = raw_word_dict.get("score", raw_word_dict.get("confidence"))
                if not word_text or start_time is None or end_time is None: continue
                start_f, end_f = float(start_time), float(end_time)
                if start_f > end_f: logger.warning(f"Wort '{word_text}' ungültige Zeiten."); continue
                word_obj = WordAlignment(word_text, start_f, end_f, float(confidence_score) if confidence_score is not None else None)
                if config.return_char_alignments and "chars" in raw_word_dict and isinstance(raw_word_dict["chars"],list):
                    word_obj.char_alignments = raw_word_dict["chars"]
                word_objects_for_segment.append(word_obj)
            segment_text = str(raw_segment_dict.get("text", "")).strip()
            if word_objects_for_segment or segment_text:
                seg_start = raw_segment_dict.get("start", word_objects_for_segment[0].start if word_objects_for_segment else 0.0)
                seg_end = raw_segment_dict.get("end", word_objects_for_segment[-1].end if word_objects_for_segment else 0.0)
                if seg_start is None or seg_end is None: continue
                final_text = segment_text if segment_text else " ".join(w.word for w in word_objects_for_segment)
                output_aligned_segments.append(AlignedSegment(float(seg_start), float(seg_end), final_text, word_objects_for_segment))
        return output_aligned_segments
    def cleanup_models(self): self.mem_cache.clear_cache()

class AlignmentService:
    def __init__(self, hardware_capabilities: HardwareCapabilities, model_cache_directory: str):
        self.engine = AlignmentEngine(hardware_capabilities, model_cache_directory)
        self.supported_languages = {"en","es","fr","de","it","pt","pl","nl","ru","ja","zh","ar","cs","el","fa","fi","he","hu","ko","sv","tr","uk","vi"}
        logger.info(f"AlignmentService initialisiert. Unterstützte Sprachen (Auswahl): {list(self.supported_languages)[:5]}")

    async def align_transcription_result(
        self, 
        transcription_result: TranscriptionResult, 
        audio: np.ndarray, # KORRIGIERTER Parametername hier
        sample_rate: int, # KORRIGIERTER Parametername hier (statt sr)
        config: Optional[AlignmentConfig]=None, # KORRIGIERTER Parametername hier (statt cfg)
        progress_callback: Optional[callable]=None # KORRIGIERTER Parametername hier (statt cb)
    ) -> AlignmentResult:
        detected_language = transcription_result.language
        if not detected_language: raise ValueError("TranscriptionResult muss Sprache für Alignment spezifizieren.")
        if detected_language not in self.supported_languages:
            logger.warning(f"Sprache '{detected_language}' nicht explizit unterstützt. Alignment könnte fehlschlagen.")
        
        effective_config = config if config else AlignmentConfig.get_precise_config(language=detected_language)
        effective_config.language_code = detected_language 

        segments_for_alignment = [{"text": seg.text, "start": seg.start, "end": seg.end} for seg in transcription_result.segments]
        
        # Stelle sicher, dass die Parameter an self.engine.align_transcription auch korrekt benannt sind
        return await self.engine.align_transcription(
            segments=segments_for_alignment, 
            audio=audio, # Parametername 'audio' hier korrekt
            sr=sample_rate, # Parametername 'sr' für engine.align_transcription
            cfg=effective_config, # Parametername 'cfg' für engine.align_transcription
            progress_callback=progress_callback # Parametername 'progress_callback' hier korrekt
        )

    async def align_raw_segments(
        self, raw_segments: List[Dict[str, Any]], audio_array: np.ndarray, sample_rate: int,
        language_code: str = "de", quality: AlignmentQuality = AlignmentQuality.PRECISE,
        progress_callback: Optional[callable] = None
    ) -> AlignmentResult:
        if language_code not in self.supported_languages: logger.warning(f"Sprache '{language_code}' nicht explizit unterstützt.")
        if quality == AlignmentQuality.BASIC: current_config = AlignmentConfig.get_basic_config(language=language_code)
        elif quality == AlignmentQuality.PHONEME: current_config = AlignmentConfig.get_phoneme_config(language=language_code)
        else: current_config = AlignmentConfig.get_precise_config(language=language_code)
        
        return await self.engine.align_transcription(
            segments=raw_segments, 
            audio=audio_array, # Hier audio_array zu audio für Konsistenz mit engine
            sr=sample_rate, 
            cfg=current_config, 
            progress_callback=progress_callback
        )

    def get_cache_info(self) -> Dict[str, Any]: # Korrekter Name
        return self.engine.mem_cache.get_cache_info()

    def clear_internal_cache(self): self.engine.cleanup_models()
    def is_language_explicitly_supported(self, l_code: str) -> bool: return l_code in self.supported_languages

# Diagnose- und Quick-Fix-Funktionen (optional, für Debugging)
# Diese sind hier nicht das Hauptaugenmerk, aber stelle sicher, dass sie bei Bedarf funktionieren
# oder kommentiere sie aus, bis die Kernprobleme gelöst sind.
def diagnose_model_cache(cache_base_str: str = "/app/.cache/huggingface") -> Dict[str, Any]:
    logger.info(f"Diagnose für Cache: {cache_base_str} (Implementierung ausgelassen für Kürze in dieser Antwort)")
    # Hier sollte die ausführliche Diagnose-Logik aus früheren Antworten stehen, wenn benötigt.
    return {"info": "Diagnose noch nicht vollständig implementiert in dieser Version für Kürze."}

async def quick_fix_alignment_model_path(cache_base_for_diag: str = "/app/.cache/huggingface") -> Optional[Path]:
    logger.info(f"Quick-Fix für Cache: {cache_base_for_diag} (Implementierung ausgelassen für Kürze in dieser Antwort)")
    # Hier sollte die ausführliche Quick-Fix-Logik aus früheren Antworten stehen, wenn benötigt.
    return None