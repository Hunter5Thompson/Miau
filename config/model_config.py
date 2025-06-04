# config/model_config.py
"""
Model Loading and Configuration Management

Abstrahiert die komplexe Model-Loading-Logik in eine saubere,
testbare Schnittstelle. Wie ein Bibliothekar, der weiß, 
wo jedes Buch steht und wie man es am besten ausleiht.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import gc

import torch
import whisperx
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum für verschiedene Model-Typen"""
    FASTER_WHISPER = "faster_whisper"
    WHISPERX = "whisperx"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Metadaten für geladene Modelle"""
    model_name: str
    model_type: ModelType
    device: str
    compute_type: str
    cache_path: Optional[str] = None
    size_mb: Optional[float] = None
    load_time_seconds: Optional[float] = None


class ModelPathResolver:
    """Resolves model paths across different cache strategies"""
    
    def __init__(self, cache_base: Path):
        self.cache_base = cache_base
        self.model_cache_dir = cache_base / "huggingface"
        self.whisper_cache_dir = cache_base / "whisper"
    
    def get_model_paths(self, model_name: str) -> List[Path]:
        """
        Gibt alle möglichen Pfade für ein Modell zurück.
        
        Args:
            model_name: Name des Modells (z.B. "large-v2")
            
        Returns:
            List[Path]: Prioritäts-sortierte Liste möglicher Pfade
        """
        model_paths = [
            # Whisper-specific cache
            self.whisper_cache_dir / f"models--Systran--faster-whisper-{model_name}",
            
            # HuggingFace cache
            self.model_cache_dir / "hub" / f"models--Systran--faster-whisper-{model_name}",
            
            # Legacy local paths (from volumes)
            Path(f"./model_cache/hub/models--Systran--faster-whisper-{model_name}"),
            Path(f"./whisper_cache/models--Systran--faster-whisper-{model_name}"),
            
            # Alternative formats
            self.model_cache_dir / f"faster-whisper-{model_name}",
            self.whisper_cache_dir / f"faster-whisper-{model_name}",
        ]
        
        return model_paths
    
    def find_existing_model(self, model_name: str) -> Optional[Path]:
        """
        Findet den ersten existierenden Model-Pfad.
        
        Args:
            model_name: Name des Modells
            
        Returns:
            Optional[Path]: Pfad zum Modell oder None
        """
        for path in self.get_model_paths(model_name):
            if path.exists() and self._validate_model_path(path):
                logger.info(f"📦 Found model at: {path}")
                return path
        
        return None
    
    def _validate_model_path(self, path: Path) -> bool:
        """Validiert, ob ein Modell-Pfad gültig ist"""
        if not path.is_dir():
            return False
        
        # Prüfe auf wichtige Model-Dateien
        required_files = ["config.json"]
        model_files = ["model.bin", "pytorch_model.bin", "model.safetensors"]
        
        has_config = any((path / f).exists() for f in required_files)
        has_model = any((path / f).exists() for f in model_files)
        
        return has_config or has_model


class ModelTypeDetector:
    """Erkennt den Typ eines geladenen Modells"""
    
    @staticmethod
    def detect_type(model_obj: Any) -> ModelType:
        """
        Erkennt den Typ eines Model-Objekts.
        
        Args:
            model_obj: Das zu analysierende Model-Objekt
            
        Returns:
            ModelType: Erkannter Modell-Typ
        """
        try:
            model_type_name = type(model_obj).__name__
            logger.debug(f"🔍 Analyzing model type: {model_type_name}")
            
            # faster-whisper Detection
            if hasattr(model_obj, 'transcribe') and hasattr(model_obj, 'feature_extractor'):
                return ModelType.FASTER_WHISPER
            
            # WhisperX Detection
            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'transcribe'):
                return ModelType.WHISPERX
            
            # Fallback tests
            if "faster_whisper" in str(type(model_obj)).lower():
                return ModelType.FASTER_WHISPER
            elif "whisperx" in str(type(model_obj)).lower():
                return ModelType.WHISPERX
            
            logger.warning(f"⚠️ Unknown model type: {type(model_obj)}")
            return ModelType.UNKNOWN
            
        except Exception as e:
            logger.error(f"❌ Model type detection failed: {e}")
            return ModelType.UNKNOWN


class ModelLoadingStrategy:
    """Base class for model loading strategies"""
    
    def load_model(self, model_name: str, device: str, compute_type: str, **kwargs) -> Any:
        """Load model with specific strategy"""
        raise NotImplementedError
    
    def get_strategy_name(self) -> str:
        """Get name of this strategy"""
        raise NotImplementedError


class FasterWhisperStrategy(ModelLoadingStrategy):
    """Loads models using faster-whisper directly"""
    
    def __init__(self, path_resolver: ModelPathResolver):
        self.path_resolver = path_resolver
    
    def load_model(self, model_name: str, device: str, compute_type: str, **kwargs) -> Any:
        """Load model using faster-whisper"""
        model_path = self.path_resolver.find_existing_model(model_name)
        
        if not model_path:
            raise FileNotFoundError(f"No faster-whisper model found for: {model_name}")
        
        logger.info(f"🎯 Loading faster-whisper model from: {model_path}")
        
        model = WhisperModel(
            str(model_path),
            device=device,
            compute_type=compute_type,
            cpu_threads=4 if device == "cpu" else 0,
            num_workers=1  # Stability for large models
        )
        
        return model
    
    def get_strategy_name(self) -> str:
        return "faster-whisper-direct"


class WhisperXStrategy(ModelLoadingStrategy):
    """Loads models using WhisperX"""
    
    def __init__(self, path_resolver: ModelPathResolver):
        self.path_resolver = path_resolver
    
    def load_model(self, model_name: str, device: str, compute_type: str, **kwargs) -> Any:
        """Load model using WhisperX"""
        logger.info(f"🔄 Loading model via WhisperX: {model_name}")
        
        # Try offline first
        old_offline = os.environ.get('HF_HUB_OFFLINE')
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type=compute_type,
                language="de",
                download_root=str(self.path_resolver.whisper_cache_dir),
                local_files_only=True
            )
            return model
        finally:
            # Restore original offline setting
            if old_offline is None:
                os.environ.pop('HF_HUB_OFFLINE', None)
            else:
                os.environ['HF_HUB_OFFLINE'] = old_offline
    
    def get_strategy_name(self) -> str:
        return "whisperx-offline"


class OnlineWhisperXStrategy(ModelLoadingStrategy):
    """Downloads models using WhisperX if offline mode disabled"""
    
    def __init__(self, path_resolver: ModelPathResolver):
        self.path_resolver = path_resolver
    
    def load_model(self, model_name: str, device: str, compute_type: str, **kwargs) -> Any:
        """Load model with online download"""
        if os.getenv("OFFLINE_MODE", "false").lower() == "true":
            raise RuntimeError("Online downloads disabled in OFFLINE_MODE")
        
        logger.info(f"🌐 Loading model with online fallback: {model_name}")
        
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language="de",
            download_root=str(self.path_resolver.whisper_cache_dir)
        )
        
        return model
    
    def get_strategy_name(self) -> str:
        return "whisperx-online"


class ModelLoader:
    """
    Zentrale Model-Loading-Klasse mit mehreren Fallback-Strategien.
    
    Funktioniert wie ein gut organisierter Taxiservice:
    - Versucht erst die schnellste Route (lokale Dateien)
    - Falls das nicht klappt, nimmt er Umwege (andere Strategien)
    - Als letztes Resort ruft er ein Taxi aus der Nachbarstadt (Online-Download)
    """
    
    def __init__(self, cache_base: Path):
        self.path_resolver = ModelPathResolver(cache_base)
        
        # Initialize loading strategies in priority order
        self.strategies = [
            FasterWhisperStrategy(self.path_resolver),
            WhisperXStrategy(self.path_resolver),
            OnlineWhisperXStrategy(self.path_resolver),
        ]
        
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, ModelMetadata] = {}
    
    def load_transcription_model(
        self, 
        model_name: str, 
        device: str, 
        compute_type: str,
        force_reload: bool = False
    ) -> Any:
        """
        Lädt Transkriptionsmodell mit mehreren Fallback-Strategien.
        
        Args:
            model_name: Name des Modells
            device: Ziel-Device (cpu/cuda)
            compute_type: Compute-Typ (float16/float32)
            force_reload: Erzwingt Neuladen auch bei gecachtem Modell
            
        Returns:
            Any: Geladenes Model-Objekt
            
        Raises:
            FileNotFoundError: Wenn alle Strategien fehlschlagen
        """
        cache_key = f"{model_name}_{device}_{compute_type}"
        
        # Check cache
        if not force_reload and cache_key in self.loaded_models:
            logger.info(f"📋 Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        logger.info(f"🔄 Loading model '{model_name}' (Device: {device}, Compute: {compute_type})")
        
        import time
        start_time = time.time()
        
        # Try each strategy
        last_error = None
        for strategy in self.strategies:
            try:
                logger.info(f"🎯 Trying strategy: {strategy.get_strategy_name()}")
                
                model = strategy.load_model(model_name, device, compute_type)
                
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded successfully with {strategy.get_strategy_name()} in {load_time:.2f}s")
                
                # Cache the model
                self.loaded_models[cache_key] = model
                
                # Store metadata  
                self.model_metadata[cache_key] = ModelMetadata(
                    model_name=model_name,
                    model_type=ModelTypeDetector.detect_type(model),
                    device=device,
                    compute_type=compute_type,
                    load_time_seconds=load_time
                )
                
                return model
                
            except Exception as e:
                logger.warning(f"⚠️ Strategy {strategy.get_strategy_name()} failed: {e}")
                last_error = e
                continue
        
        # All strategies failed
        available_models = self._list_available_models()
        error_msg = f"❌ All loading strategies failed for '{model_name}'"
        if available_models:
            error_msg += f" Available models: {available_models}"
        
        if last_error:
            error_msg += f" Last error: {last_error}"
        
        raise FileNotFoundError(error_msg)
    
    def unload_model(self, model_name: str, device: str, compute_type: str) -> bool:
        """
        Entlädt ein Modell aus dem Speicher.
        
        Args:
            model_name: Name des Modells
            device: Device des Modells
            compute_type: Compute-Typ des Modells
            
        Returns:
            bool: True wenn erfolgreich entladen
        """
        cache_key = f"{model_name}_{device}_{compute_type}"
        
        if cache_key in self.loaded_models:
            try:
                del self.loaded_models[cache_key]
                if cache_key in self.model_metadata:
                    del self.model_metadata[cache_key]
                
                # Cleanup memory
                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"🗑️ Model unloaded: {cache_key}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to unload model {cache_key}: {e}")
                return False
        
        return False
    
    def cleanup_all_models(self) -> None:
        """Entlädt alle geladenen Modelle"""
        for cache_key in list(self.loaded_models.keys()):
            try:
                del self.loaded_models[cache_key]
            except Exception as e:
                logger.warning(f"⚠️ Error unloading {cache_key}: {e}")
        
        self.loaded_models.clear()
        self.model_metadata.clear()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 All models unloaded")
    
    def get_model_info(self, model_name: str, device: str, compute_type: str) -> Optional[ModelMetadata]:
        """Gibt Metadaten für ein geladenes Modell zurück"""
        cache_key = f"{model_name}_{device}_{compute_type}"
        return self.model_metadata.get(cache_key)
    
    def list_loaded_models(self) -> Dict[str, ModelMetadata]:
        """Gibt alle geladenen Modelle zurück"""
        return self.model_metadata.copy()
    
    def _list_available_models(self) -> List[str]:
        """Listet verfügbare Modelle in Cache-Verzeichnissen auf"""
        available = []
        cache_dirs = [
            self.path_resolver.whisper_cache_dir, 
            self.path_resolver.model_cache_dir / "hub"
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                try:
                    models = [
                        d.name for d in cache_dir.iterdir() 
                        if d.is_dir() and "whisper" in d.name.lower()
                    ]
                    available.extend(models)
                except Exception:
                    pass
        
        return list(set(available))


# Default model loader instance (can be overridden)
_default_cache_base = Path(os.getenv("HF_HOME_OVERRIDE", "/app/.cache"))
default_model_loader = ModelLoader(_default_cache_base)


def load_model(model_name: str, device: str, compute_type: str, **kwargs) -> Any:
    """Convenience function for model loading"""
    return default_model_loader.load_transcription_model(model_name, device, compute_type, **kwargs)


def cleanup_models() -> None:
    """Convenience function for model cleanup"""
    default_model_loader.cleanup_all_models()