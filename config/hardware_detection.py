# config/hardware_detection.py
"""
Hardware Detection & GPU Configuration Module

Separiert die Hardware-Erkennung von der Business-Logic.
Wie ein Mechaniker, der erst mal unter die Motorhaube schaut,
bevor er den Reparaturplan macht.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Container für Hardware-Fähigkeiten"""
    device: str
    compute_type: str
    max_batch_size: int
    max_concurrent_jobs: int
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None


class CuDNNConfigurator:
    """Verwaltet cuDNN-Setup für verschiedene GPU-Generationen"""
    
    @staticmethod
    def setup_environment() -> bool:
        """
        Konfiguriert cuDNN-Umgebung für optimale Performance.
        
        Returns:
            bool: True wenn erfolgreich konfiguriert
        """
        try:
            cudnn_paths = [
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu", 
                "/opt/conda/lib",
                "/usr/local/lib"
            ]
            
            found_cudnn = False
            for path in cudnn_paths:
                if Path(path).exists():
                    cudnn_files = list(Path(path).glob("libcudnn*.so*"))
                    if cudnn_files:
                        logger.info(f"✅ cuDNN found in: {path}")
                        CuDNNConfigurator._add_to_ld_library_path(path)
                        found_cudnn = True
                        break
            
            if not found_cudnn:
                logger.warning("⚠️ No cuDNN libraries found. GPU performance may be impacted.")
                return False
                
            # Configure PyTorch cuDNN settings
            if torch.cuda.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True  # Optimized for H100
                torch.backends.cudnn.deterministic = False  # Better performance
                logger.info("🚀 PyTorch cuDNN optimizations activated")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ cuDNN setup failed: {e}")
            return False
        
        return False
    
    @staticmethod
    def _add_to_ld_library_path(path: str) -> None:
        """Fügt Pfad zu LD_LIBRARY_PATH hinzu"""
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if path not in current_ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{current_ld_path}"


class GPUDetector:
    """Erkennt und kategorisiert verfügbare GPU-Hardware"""
    
    # GPU-Kategorien mit ihren Leistungsprofilen
    GPU_PROFILES = {
        "H100": {"batch_size": 64, "concurrent_jobs": 4, "compute_type": "float16"},
        "L40S": {"batch_size": 64, "concurrent_jobs": 4, "compute_type": "float16"},
        "A100": {"batch_size": 48, "concurrent_jobs": 3, "compute_type": "float16"},
        "RTX_4090": {"batch_size": 32, "concurrent_jobs": 2, "compute_type": "float16"},
        "RTX_3090": {"batch_size": 24, "concurrent_jobs": 2, "compute_type": "float16"},
        "DEFAULT_HIGH": {"batch_size": 32, "concurrent_jobs": 2, "compute_type": "float16"},
        "DEFAULT_MID": {"batch_size": 16, "concurrent_jobs": 1, "compute_type": "float16"},
        "DEFAULT_LOW": {"batch_size": 8, "concurrent_jobs": 1, "compute_type": "float16"},
    }
    
    @classmethod
    def detect_capabilities(cls) -> HardwareCapabilities:
        """
        Erkennt GPU-Fähigkeiten mit robustem Fallback.
        
        Returns:
            HardwareCapabilities: Detected hardware configuration
        """
        if not torch.cuda.is_available():
            logger.info("🖥️ No CUDA available, using CPU configuration")
            return cls._get_cpu_capabilities()
        
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            gpu_name = gpu_props.name
            
            logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            
            # Bestimme GPU-Profil
            profile = cls._classify_gpu(gpu_name, gpu_memory_gb)
            config = cls.GPU_PROFILES[profile]
            
            return HardwareCapabilities(
                device="cuda",
                compute_type=config["compute_type"],
                max_batch_size=config["batch_size"],
                max_concurrent_jobs=config["concurrent_jobs"],
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_memory_gb,
                cuda_version=torch.version.cuda
            )
            
        except Exception as e:
            logger.warning(f"⚠️ GPU detection failed: {e}. Falling back to conservative settings.")
            return HardwareCapabilities(
                device="cuda",
                compute_type="float16",
                max_batch_size=16,
                max_concurrent_jobs=1
            )
    
    @classmethod
    def _classify_gpu(cls, gpu_name: str, memory_gb: float) -> str:
        """Klassifiziert GPU basierend auf Name und Speicher"""
        gpu_name_upper = gpu_name.upper()
        
        # Spezifische GPU-Modelle
        if "H100" in gpu_name_upper:
            return "H100"
        elif "L40S" in gpu_name_upper:
            return "L40S"
        elif "A100" in gpu_name_upper:
            return "A100"
        elif "RTX 4090" in gpu_name_upper or "4090" in gpu_name_upper:
            return "RTX_4090"
        elif "RTX 3090" in gpu_name_upper or "3090" in gpu_name_upper:
            return "RTX_3090"
        
        # Fallback basierend auf Speicher
        if memory_gb >= 70:
            return "DEFAULT_HIGH"
        elif memory_gb >= 20:
            return "DEFAULT_MID"
        else:
            return "DEFAULT_LOW"
    
    @classmethod
    def _get_cpu_capabilities(cls) -> HardwareCapabilities:
        """Gibt CPU-basierte Konfiguration zurück"""
        return HardwareCapabilities(
            device="cpu",
            compute_type="float32",
            max_batch_size=8,
            max_concurrent_jobs=1
        )


class H100Optimizer:
    """Spezielle Optimierungen für H100-GPUs"""
    
    @staticmethod
    def setup_memory_optimization() -> bool:
        """
        Optimiert PyTorch Memory-Management für H100.
        
        Returns:
            bool: True wenn erfolgreich optimiert
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            torch.cuda.empty_cache()
            
            # H100-spezifische Memory-Konfiguration
            alloc_conf = "max_split_size_mb:1024,roundup_power2_divisions:16"
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = alloc_conf
            logger.info(f"✅ H100 memory optimization activated: {alloc_conf}")
            
            # TensorFloat-32 Optimierungen für H100
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            logger.info("🚀 H100 TF32 optimizations enabled")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ H100 memory optimization failed: {e}")
            return False


class HardwareManager:
    """
    Zentrale Hardware-Management-Klasse.
    
    Der Dirigent des Hardware-Orchesters - koordiniert alle 
    Hardware-bezogenen Operationen an einem Ort.
    """
    
    def __init__(self):
        self._capabilities: Optional[HardwareCapabilities] = None
        self._initialized = False
    
    def initialize(self) -> HardwareCapabilities:
        """
        Initialisiert Hardware-Erkennung und -Optimierung.
        
        Returns:
            HardwareCapabilities: Detected and optimized hardware config
        """
        if self._initialized and self._capabilities:
            return self._capabilities
        
        logger.info("🔧 Initializing hardware detection...")
        
        # 1. GPU Detection
        self._capabilities = GPUDetector.detect_capabilities()
        
        # 2. cuDNN Setup (nur bei GPU)
        if self._capabilities.device == "cuda":
            CuDNNConfigurator.setup_environment()
            
            # 3. H100-spezifische Optimierungen
            if self._capabilities.gpu_name and "H100" in self._capabilities.gpu_name:
                H100Optimizer.setup_memory_optimization()
        
        self._initialized = True
        
        logger.info(
            f"✅ Hardware initialized: {self._capabilities.device}/"
            f"{self._capabilities.compute_type}, "
            f"Batch={self._capabilities.max_batch_size}, "
            f"Concurrent={self._capabilities.max_concurrent_jobs}"
        )
        
        return self._capabilities
    
    @property
    def capabilities(self) -> HardwareCapabilities:
        """Gibt Hardware-Capabilities zurück (lazy initialization)"""
        if not self._initialized:
            return self.initialize()
        return self._capabilities
    
    def get_transcription_params(self) -> Dict[str, Any]:
        """
        Gibt optimierte Transkriptionsparameter basierend auf Hardware zurück.
        
        Returns:
            Dict: Optimized transcription parameters
        """
        caps = self.capabilities
        
        if caps.device == "cuda" and caps.gpu_name and "H100" in caps.gpu_name:
            logger.info("💡 Using H100 high-performance parameters")
            return {
                "language": "de",
                "beam_size": 5,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "temperature": [0.0, 0.2, 0.4],  # Multiple temperatures for quality
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "prepend_punctuations": "\"'¿([{-",
                "append_punctuations": "\"'.。,，!！?？:：)]}、"
            }
        
        # Standard parameters for other hardware
        return {
            "language": "de",
            "beam_size": 3,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "temperature": 0.0
        }
    
    def cleanup_resources(self) -> None:
        """Bereinigt Hardware-Ressourcen"""
        if self._capabilities and self._capabilities.device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.info("🧹 GPU memory cleared")
            except Exception as e:
                logger.warning(f"⚠️ GPU cleanup failed: {e}")


# Singleton-Instance für globale Verwendung
hardware_manager = HardwareManager()


def get_hardware_capabilities() -> HardwareCapabilities:
    """Convenience function für Hardware-Capabilities"""
    return hardware_manager.capabilities


def initialize_hardware() -> HardwareCapabilities:
    """Convenience function für Hardware-Initialisierung"""
    return hardware_manager.initialize()