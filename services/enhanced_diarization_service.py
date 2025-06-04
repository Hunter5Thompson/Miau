# services/enhanced_diarization_service.py
"""
Enhanced Diarization Service mit Nexus-Support

Erweitert den bestehenden DiarizationService um Nexus-Repository-Zugriff
für PyAnnote Audio Models.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from services.diarization_service import DiarizationService
from config.hardware_detection import HardwareCapabilities

logger = logging.getLogger(__name__)


class NexusDiarizationEngine:
    """
    Wrapper für PyAnnote-Models mit Nexus-Fallback.
    
    Wie ein diplomatischer Übersetzer, der sowohl lokale als auch
    internationale Dokumentenarchive durchsuchen kann.
    """
    
    def __init__(self, hf_token: Optional[str], nexus_base_url: Optional[str] = None):
        self.hf_token = hf_token
        self.nexus_base_url = nexus_base_url
        self.fallback_models = {
            "default": "pyannote/speaker-diarization-3.1",
            "segmentation": "pyannote/segmentation-3.0", 
            "embedding": "pyannote/wespeaker-voxceleb-resnet34-LM"
        }
    
    def setup_nexus_environment(self) -> bool:
        """
        Konfiguriert Environment für Nexus-Repository-Zugriff.
        
        Returns:
            bool: True wenn Nexus verfügbar
        """
        if not self.nexus_base_url:
            return False
            
        try:
            # Test Nexus-Connectivity
            import requests
            response = requests.head(f"{self.nexus_base_url}/pyannote/speaker-diarization-3.1", timeout=10)
            
            if response.status_code == 200:
                logger.info("🌐 Nexus PyAnnote repository verfügbar")
                
                # Environment für HuggingFace Hub konfigurieren
                os.environ["HF_ENDPOINT"] = self.nexus_base_url
                # Alternativ: Custom Hub-URL setzen falls unterstützt
                
                return True
            else:
                logger.warning(f"⚠️ Nexus PyAnnote nicht erreichbar: {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Nexus-Test fehlgeschlagen: {e}")
            return False
    
    def download_diarization_models(self, cache_dir: Path) -> bool:
        """
        Lädt Diarisierung-Modelle von Nexus herunter.
        
        Args:
            cache_dir: Ziel-Cache-Verzeichnis
            
        Returns:
            bool: Success status
        """
        if not self.nexus_base_url:
            return False
            
        success_count = 0
        
        for model_name, model_repo in self.fallback_models.items():
            try:
                target_dir = cache_dir / "hub" / f"models--{model_repo.replace('/', '--')}"
                
                if target_dir.exists():
                    logger.info(f"📋 Diarization model bereits cached: {model_name}")
                    success_count += 1
                    continue
                
                logger.info(f"🌐 Lade Diarization-Modell von Nexus: {model_repo}")
                
                if self._download_pyannote_model(model_repo, target_dir):
                    success_count += 1
                    logger.info(f"✅ Diarization-Modell geladen: {model_name}")
                else:
                    logger.error(f"❌ Download fehlgeschlagen: {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Fehler beim Laden von {model_name}: {e}")
        
        return success_count > 0
    
    def _download_pyannote_model(self, model_repo: str, target_dir: Path) -> bool:
        """Downloads PyAnnote model from Nexus"""
        try:
            import requests
            from pathlib import Path
            
            # PyAnnote-spezifische Dateien
            required_files = [
                "config.yaml",
                "pytorch_model.bin",
                "model.safetensors"  # Alternative zu pytorch_model.bin
            ]
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            downloaded_files = 0
            for filename in required_files:
                file_url = f"{self.nexus_base_url}/{model_repo}/{filename}"
                file_path = target_dir / filename
                
                try:
                    response = requests.get(file_url, timeout=30)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        downloaded_files += 1
                        logger.debug(f"✅ Downloaded: {filename}")
                    else:
                        logger.debug(f"⚠️ File not found: {filename} (HTTP {response.status_code})")
                        
                except Exception as e:
                    logger.debug(f"⚠️ Download error for {filename}: {e}")
            
            # Mindestens eine Datei muss erfolgreich geladen worden sein
            return downloaded_files > 0
            
        except Exception as e:
            logger.error(f"❌ PyAnnote model download failed: {e}")
            return False


class EnhancedDiarizationService(DiarizationService):
    """
    Enhanced Diarization Service mit Nexus-Integration.
    
    Erweitert den Standard-DiarizationService um Nexus-Repository-Zugriff
    ohne bestehende Funktionalität zu brechen.
    """
    
    def __init__(self, hardware_caps: HardwareCapabilities, hf_token: Optional[str] = None):
        # Initialize base service
        super().__init__(hardware_caps, hf_token)
        
        # Nexus integration
        nexus_url = os.getenv("NEXUS_HUGGINGFACE_URL")
        self.nexus_engine = NexusDiarizationEngine(hf_token, nexus_url) if nexus_url else None
        
        # Enhanced availability check
        self._check_enhanced_availability()
    
    def _check_enhanced_availability(self) -> None:
        """Enhanced availability check mit Nexus-Fallback"""
        if self.is_available:
            logger.info("✅ Diarization: Standard HuggingFace verfügbar")
            return
        
        if self.nexus_engine and self.nexus_engine.setup_nexus_environment():
            logger.info("🌐 Diarization: Nexus-Fallback verfügbar")
            self.is_available = True
            
            # Pre-download models if not cached
            cache_dir = Path(os.getenv("HF_HOME", "/app/.cache/huggingface"))
            if self.nexus_engine.download_diarization_models(cache_dir):
                logger.info("📦 Diarization-Modelle von Nexus geladen")
            else:
                logger.warning("⚠️ Nexus-Download teilweise fehlgeschlagen")
        else:
            logger.warning("❌ Diarization nicht verfügbar (weder HF noch Nexus)")
    
    async def diarize_transcription_result(self, *args, **kwargs):
        """
        Enhanced diarization mit automatischem Nexus-Fallback.
        
        Versucht zuerst Standard-HuggingFace, dann Nexus-Repository.
        """
        try:
            # First try: Standard implementation
            return await super().diarize_transcription_result(*args, **kwargs)
            
        except Exception as e:
            if "403" in str(e) or "authentication" in str(e).lower():
                logger.warning("🔐 HuggingFace authentication issue, trying Nexus fallback...")
                
                if self.nexus_engine:
                    # Setup Nexus environment and retry
                    if self.nexus_engine.setup_nexus_environment():
                        logger.info("🌐 Retrying with Nexus configuration...")
                        return await super().diarize_transcription_result(*args, **kwargs)
                
            # Re-raise if no fallback worked
            raise e


# Factory function für Service-Erstellung
def create_enhanced_diarization_service(
    hardware_caps: HardwareCapabilities, 
    hf_token: Optional[str] = None
) -> EnhancedDiarizationService:
    """
    Factory function für Enhanced Diarization Service.
    
    Args:
        hardware_caps: Hardware capabilities
        hf_token: HuggingFace token
        
    Returns:
        EnhancedDiarizationService: Configured service
    """
    service = EnhancedDiarizationService(hardware_caps, hf_token)
    
    if service.is_available:
        logger.info("✅ Enhanced Diarization Service bereit")
    else:
        logger.warning("⚠️ Enhanced Diarization Service nicht verfügbar")
    
    return service