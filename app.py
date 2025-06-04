# app.py - FULLY REFACTORED VERSION (mit Frontend-Integration)
"""
WhisperX Protokoll-Service - COMPLETE REFACTORED VERSION

Von einem 1000-Zeilen-Spaghetti-Monster zu einem sauberen,
modularen Service mit klaren Verantwortlichkeiten.
"""

import asyncio
import uuid
import shutil
import time
import re
import json
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

# ===== REFACTORED SERVICE IMPORTS =====
# Hardware & Configuration
from config.hardware_detection import initialize_hardware, get_hardware_capabilities
from config.nexus_integration import create_enhanced_model_loader
enhanced_model_loader = create_enhanced_model_loader()
from config import app_settings
settings = app_settings.initialize_settings()
from config.app_settings import initialize_settings, get_settings
settings = app_settings.initialize_settings() # Sicherstellen, dass settings hier initialisiert wird

# Audio Services
from services.audio_service import AudioService, AudioValidationError
from services.audio_preprocessing import (
    AudioPreprocessor, PreprocessingConfig,
    preprocess_for_speech, preprocess_minimal
)

# ML Services
from services.ml_pipeline import (
    MLPipelineOrchestrator, PipelineConfig, PipelineMode, PipelineProgress
)

# FastAPI
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles # Bereits vorhanden, aber zur Verdeutlichung hier
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel, Field
import os
import logging

# Imports für Frontend-Integration (bereits vorhanden, hier zur Bestätigung)
from fastapi.responses import HTMLResponse

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.refactored_transcription_service")

# ===== INITIALIZATION PHASE =====
logger.info("🚀 Initializing Refactored WhisperX Service...")

# 1. Initialize settings (creates directories, sets env vars)
# settings = initialize_settings() # Bereits oben initialisiert, um Modul-Level Fehler zu vermeiden
logger.info(f"⚙️ Settings initialized: {settings.cache_base}")

# 2. Initialize hardware detection (GPU/CPU, cuDNN, optimizations)
hardware_caps = initialize_hardware()
logger.info(f"🎮 Hardware initialized: {hardware_caps.device}")

# 3. Initialize all services
audio_service = AudioService(
    max_duration=settings.max_audio_duration,
    min_sample_rate=settings.min_sample_rate,
    max_file_size=settings.max_file_size,
    supported_formats=settings.supported_audio_formats
)

ml_pipeline = MLPipelineOrchestrator(
    hardware_caps=hardware_caps,
    model_loader=enhanced_model_loader, # <- Hier der Enhanced Loader
    model_cache_dir=str(settings.model_cache_dir),
    hf_token=settings.huggingface_token
)

logger.info("✅ All services initialized successfully")

# ===== GLOBAL STATE MANAGEMENT =====
# Thread-safe Job-Management (will be refactored in Business Logic phase)
jobs: Dict[str, "TranscriptionJob"] = {}
jobs_lock = Lock()

# ===== PYDANTIC MODELS =====
class TranscriptionJob(BaseModel):
    job_id: str
    file_name: str
    municipality: str
    meeting_date: str
    meeting_type: str
    status: str = Field(..., description="Status (pending, processing, completed, failed)")
    progress_stage: Optional[str] = Field(None, description="Current processing stage")
    progress_percent: Optional[float] = Field(None, description="Progress 0.0-100.0")
    created_at: str
    completed_at: Optional[str] = None
    download_url: Optional[str] = None

    # Enhanced fields for refactored version
    pipeline_mode: Optional[str] = Field(None, description="Pipeline processing mode")
    preprocessing_mode: Optional[str] = Field(None, description="Audio preprocessing mode")
    estimated_time: Optional[float] = Field(None, description="Estimated processing time")

    model_config = {'protected_namespaces': ()}

class TranscriptionResult(BaseModel):
    job_id: str
    file_name: str
    municipality: str
    meeting_date: str
    meeting_type: str
    transcription_text: str
    processing_time: float
    download_url: str

    # Enhanced fields
    pipeline_statistics: Optional[Dict[str, Any]] = None
    quality_metrics: Optional[Dict[str, Any]] = None

# ===== JOB MANAGEMENT FUNCTIONS =====
# (These will be moved to job_manager.py in Business Logic phase)
def get_job(job_id: str) -> Optional[TranscriptionJob]:
    with jobs_lock:
        return jobs.get(job_id)

def update_job(job_id: str, **kwargs: Any) -> bool:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return False

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        return True

def add_job(job: TranscriptionJob) -> None:
    with jobs_lock:
        jobs[job.job_id] = job
        logger.info(f"➕ Job {job.job_id} added")

def list_all_jobs() -> List[TranscriptionJob]:
    with jobs_lock:
        return list(jobs.values())

def remove_job(job_id: str) -> Optional[TranscriptionJob]:
    with jobs_lock:
        return jobs.pop(job_id, None)

# ===== FASTAPI APP SETUP =====
app = FastAPI(
    title="Refactored Gemeinderatsprotokolle-Transkription",
    description="Clean, modular API for automated transcription with ML services",
    version="3.0.0-refactored"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In Produktion spezifischer setzen!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static file serving
app.mount("/downloads", StaticFiles(directory=str(settings.protocols_dir)), name="downloads")
# HINZUGEFÜGT/BESTÄTIGT: Frontend static files
frontend_dir = Path("frontend")
if frontend_dir.exists() and frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
    logger.info("✅ Frontend mounted at /static")
else:
    logger.warning("⚠️ Frontend directory not found - running API-only mode")


# ===== STARTUP/SHUTDOWN EVENTS =====
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Refactored WhisperX Service starting...")

    # Display service information
    logger.info(f"🎮 Hardware: {hardware_caps.device}")
    if hardware_caps.gpu_name:
        logger.info(f" GPU: {hardware_caps.gpu_name} ({hardware_caps.gpu_memory_gb:.1f}GB)")
    if hardware_caps.cuda_version:
        logger.info(f" CUDA: {hardware_caps.cuda_version}")

    # Test model availability
    try:
        available_models = enhanced_model_loader._list_available_models() # Zugriff auf private Methode, ggf. anpassen
        if available_models:
            logger.info(f"📦 Available models: {len(available_models)}")
        else:
            logger.warning("⚠️ No local models found - will download on demand")
    except Exception as e:
        logger.error(f"❌ Model check failed: {e}")

    logger.info(f"✅ Refactored service ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown with service cleanup"""
    logger.info("🛑 Refactored service shutting down...")

    # Wait for active jobs
    active_jobs = [j for j in list_all_jobs() if j.status == "processing"]
    if active_jobs:
        logger.info(f"⏳ Waiting for {len(active_jobs)} active jobs...")

        wait_start = time.time()
        while active_jobs and (time.time() - wait_start < 30):
            await asyncio.sleep(1)
            active_jobs = [j for j in list_all_jobs() if j.status == "processing"]

        if active_jobs:
            logger.warning(f"⚠️ {len(active_jobs)} jobs still active during shutdown")

    # Clean up ML services
    ml_pipeline.cleanup_resources()

    logger.info("✅ Refactored service shutdown complete")

# ===== FRONTEND SERVING ENDPOINT =====
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the main frontend HTML file."""
    frontend_index_path = Path("frontend") / "index.html"
    if not frontend_index_path.is_file():
        logger.error(f"Frontend index.html not found at {frontend_index_path}")
        raise HTTPException(status_code=500, detail="Frontend not found. Contact administrator.")
    with open(frontend_index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# ===== API INFO ENDPOINT (Previously Root) =====
@app.get("/api/", tags=["Service Info"]) # Tags für bessere Swagger-UI Organisation
async def api_info():
    """Enhanced root endpoint with detailed service information (moved from /)"""
    return {
        "message": "Refactored H100-optimized Transcription API",
        "version": app.version,
        "architecture": "microservices", # Oder was auch immer zutrifft
        "services": {
            "audio_service": {
                "supported_formats": audio_service.get_supported_formats(),
                "max_file_size_mb": settings.max_file_size / (1024 * 1024),
                "max_duration_hours": settings.max_audio_duration / 3600
            },
            "ml_pipeline": {
                "transcription": True,
                "alignment": True,
                "diarization": ml_pipeline.diarization_service.is_available,
                "protocol_formatting": True # Annahme, dass dies immer verfügbar ist
            }
        },
        "hardware": {
            "device": hardware_caps.device,
            "gpu_name": hardware_caps.gpu_name,
            "gpu_memory_gb": hardware_caps.gpu_memory_gb,
            "max_concurrent_jobs": hardware_caps.max_concurrent_jobs,
            "max_batch_size": hardware_caps.max_batch_size
        },
        "configuration": {
            "default_model": settings.default_model,
            "offline_mode": settings.offline_mode,
            "cache_base": str(settings.cache_base)
        }
    }

@app.get("/health", tags=["Service Info"])
async def health_check():
    """Enhanced health check with detailed service status"""
    active_jobs = len([j for j in list_all_jobs() if j.status == "processing"])

    # Check service health
    services_status = {
        "audio_service": "healthy",
        "transcription_service": "healthy",
        "alignment_service": "healthy",
        "diarization_service": "healthy" if ml_pipeline.diarization_service.is_available else "unavailable",
        "protocol_service": "healthy" # Annahme
    }

    overall_status = "healthy" if all(
        status in ["healthy", "unavailable"] for status in services_status.values()
    ) else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "device": hardware_caps.device,
            "gpu_available": hardware_caps.device == "cuda"
        },
        "services": services_status,
        "jobs": {
            "active": active_jobs,
            "max_concurrent": hardware_caps.max_concurrent_jobs
        },
        "cache": {
            "model_cache_size": len(enhanced_model_loader.list_loaded_models()),
            "alignment_cache_size": len(ml_pipeline.alignment_service.get_cache_info().get("cached_models", []))
        }
    }

@app.get("/pipeline/recommendations", tags=["Pipeline"])
async def get_pipeline_recommendations(
    audio_duration: float,
    expected_speakers: Optional[int] = None,
    meeting_type: str = "Stadtrat",
    quality_priority: str = "balanced"
):
    """Get pipeline configuration recommendations"""
    recommendations = ml_pipeline.get_pipeline_recommendations(
        audio_duration=audio_duration,
        expected_speakers=expected_speakers,
        meeting_type=meeting_type,
        quality_priority=quality_priority
    )

    return {
        "recommendations": recommendations,
        "available_modes": [mode.value for mode in PipelineMode],
        "quality_priorities": ["speed", "balanced", "quality"]
    }

# ===== MAIN TRANSCRIPTION ENDPOINT =====
@app.post("/transcribe/", response_model=TranscriptionJob, tags=["Transcription"])
async def transcribe_audio_refactored(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    municipality: str = Form(...),
    meeting_date: str = Form(...),
    meeting_type: str = Form("Stadtrat"),
    model_name: str = Form(settings.default_model),
    pipeline_mode: str = Form("balanced"),
    preprocessing_mode: str = Form("speech"),
    enable_diarization: bool = Form(True),
    enable_alignment: bool = Form(True),
    expected_speakers: Optional[int] = Form(None)
):
    """
    FULLY REFACTORED transcription endpoint with clean service orchestration.
    """

    # === INPUT VALIDATION ===
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    # Validate pipeline mode
    valid_modes = [mode.value for mode in PipelineMode if mode != PipelineMode.CUSTOM]
    if pipeline_mode not in valid_modes:
        raise HTTPException(400, f"Invalid pipeline mode. Valid modes: {valid_modes}")

    # Validate preprocessing mode
    valid_preprocessing = ["speech", "minimal", "off"]
    if preprocessing_mode not in valid_preprocessing:
        raise HTTPException(400, f"Invalid preprocessing mode. Valid modes: {valid_preprocessing}")

    # File format validation using audio service
    extension = Path(file.filename).suffix.lower()
    if not audio_service.is_supported_format(extension):
        supported = audio_service.get_supported_formats()
        raise HTTPException(400, f"Format {extension} not supported. Supported: {supported}")

    # Date validation
    try:
        datetime.strptime(meeting_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format (YYYY-MM-DD required)")

    # === FILE UPLOAD ===
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{extension}"
    upload_path = settings.upload_dir / safe_filename

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📁 File uploaded: {upload_path.name}")
    except Exception as e:
        raise HTTPException(500, f"Upload error: {e}")
    finally:
        if file.file: # Check if file object exists before trying to close
            try:
                await asyncio.to_thread(file.file.close)
            except Exception as close_exc:
                logger.warning(f"Could not close uploaded file: {close_exc}")


    # === AUDIO VALIDATION ===
    try:
        is_valid, message, metadata = await audio_service.validate_audio_file(
            upload_path, file.filename
        )

        if not is_valid:
            if upload_path.exists():
                await asyncio.to_thread(upload_path.unlink, True)
            raise HTTPException(400, f"Invalid audio: {message}")

        logger.info(
            f"✅ Audio validated: {metadata.duration_seconds:.2f}s, "
            f"{metadata.sample_rate}Hz, {metadata.format.upper()}"
        )

    except AudioValidationError as e:
        if upload_path.exists():
            await asyncio.to_thread(upload_path.unlink, True)
        raise HTTPException(400, f"Audio validation failed: {e}")

    # === GET PROCESSING RECOMMENDATIONS ===
    recommendations = ml_pipeline.get_pipeline_recommendations(
        audio_duration=metadata.duration_seconds,
        expected_speakers=expected_speakers,
        meeting_type=meeting_type,
        quality_priority="balanced" if pipeline_mode == "balanced" else
                         "speed" if pipeline_mode == "fast" else "quality"
    )

    estimated_time = recommendations.get("estimated_processing_time_minutes", 5.0)

    # === JOB CREATION ===
    job = TranscriptionJob(
        job_id=job_id,
        file_name=file.filename,
        municipality=municipality,
        meeting_date=meeting_date,
        meeting_type=meeting_type,
        status="pending",
        progress_stage="queued",
        progress_percent=0.0,
        created_at=datetime.now().isoformat(),
        pipeline_mode=pipeline_mode,
        preprocessing_mode=preprocessing_mode,
        estimated_time=estimated_time
    )
    add_job(job)

    # === BACKGROUND PROCESSING ===
    background_tasks.add_task(
        process_with_ml_pipeline,
        job_id, upload_path, file.filename, metadata,
        municipality, meeting_date, meeting_type,
        model_name, pipeline_mode, preprocessing_mode,
        enable_diarization, enable_alignment, expected_speakers
    )

    return job

# ===== REFACTORED PROCESSING FUNCTION =====
async def process_with_ml_pipeline(
    job_id: str,
    file_path: Path,
    original_file_name: str,
    audio_metadata, # Dies ist ein AudioMetadata Objekt von audio_service
    municipality: str,
    meeting_date: str,
    meeting_type: str,
    model_name: str,
    pipeline_mode: str,
    preprocessing_mode: str,
    enable_diarization: bool,
    enable_alignment: bool,
    expected_speakers: Optional[int]
):
    """
    FULLY REFACTORED processing function using clean ML pipeline orchestration.
    """

    start_time = time.time()

    # Progress callback to update job status
    def progress_callback(progress: PipelineProgress):
        try:
            update_job(
                job_id,
                progress_stage=progress.stage,
                progress_percent=progress.progress * 100,
                status="processing"
            )
            logger.info(
                f"📊 Job {job_id}: {progress.stage} ({progress.progress*100:.0f}%) - "
                f"{progress.message}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Progress callback error: {e}")

    try:
        update_job(job_id, status="processing", progress_stage="initializing", progress_percent=0.0)
        logger.info(f"🚀 Processing with ML pipeline: {job_id} ({original_file_name})")

        # === STEP 1: LOAD AND PREPROCESS AUDIO ===
        progress_callback(PipelineProgress("loading_audio", 0.05, "Loading audio", time.time(), start_time))

        audio_data, _ = await audio_service.validate_and_load_audio(
            file_path=file_path,
            original_filename=original_file_name,
            sr=16000, # WhisperX standard
            mono=True
        )

        # Apply preprocessing if requested
        if preprocessing_mode == "speech":
            logger.info("🎛️ Applying speech-optimized preprocessing")
            preprocessing_result = preprocess_for_speech(audio_data, 16000)
            final_audio = preprocessing_result.processed_audio
        elif preprocessing_mode == "minimal":
            logger.info("🎛️ Applying minimal preprocessing")
            preprocessing_result = preprocess_minimal(audio_data, 16000)
            final_audio = preprocessing_result.processed_audio
        else: # "off"
            logger.info("🎛️ Skipping preprocessing")
            final_audio = audio_data

        # === STEP 2: CREATE PIPELINE CONFIGURATION ===
        progress_callback(PipelineProgress("configuring_pipeline", 0.1, "Configuring ML pipeline", time.time(), start_time))

        # Create pipeline configuration based on mode
        if pipeline_mode == "fast":
            config = PipelineConfig.get_fast_config()
        elif pipeline_mode == "accurate":
            config = PipelineConfig.get_accurate_config(hardware_caps)
        elif pipeline_mode == "premium":
            config = PipelineConfig.get_premium_config(hardware_caps)
        else: # balanced
            config = PipelineConfig.get_balanced_config()

        # Override with user preferences
        config.transcription_model = model_name
        config.enable_alignment = enable_alignment
        config.enable_diarization = enable_diarization and ml_pipeline.diarization_service.is_available

        # Adjust diarization config for expected speakers
        if expected_speakers and config.diarization_config:
            config.diarization_config.max_speakers = min(expected_speakers + 2, 12)
            config.diarization_config.min_speakers = max(expected_speakers - 1, 2) if expected_speakers > 2 else None

        # === STEP 3: RUN ML PIPELINE ===
        logger.info(f"🤖 Starting ML pipeline: {config.mode.value} mode")

        pipeline_result = await ml_pipeline.process_audio(
            audio=final_audio,
            sample_rate=16000,
            municipality=municipality,
            meeting_date=meeting_date,
            meeting_type=meeting_type,
            config=config,
            progress_callback=progress_callback
        )

        # === STEP 4: SAVE RESULTS ===
        progress_callback(PipelineProgress("saving_results", 0.95, "Saving results", time.time(), start_time))

        # Save protocol if available
        download_url = None
        if pipeline_result.protocol_result:
            safe_municipality = re.sub(r'\W+', '_', municipality)
            protocol_filename = f"{job_id}_{safe_municipality}_{meeting_date}.txt"
            protocol_path = settings.protocols_dir / protocol_filename

            pipeline_result.protocol_result.save_to_file(protocol_path)
            download_url = f"/downloads/{protocol_filename}"

            logger.info(f"📄 Protocol saved: {protocol_filename}")

        # Save detailed results as JSON
        # final_result = pipeline_result.get_final_result() # Nicht direkt verwendet

        # Get text from the most processed result
        if pipeline_result.protocol_result:
            full_text = pipeline_result.protocol_result.formatted_text
        elif pipeline_result.diarization_result:
            full_text = "\n".join([
                f"{seg.speaker_id}: {seg.text}"
                for seg in pipeline_result.diarization_result.segments
            ])
        else: # Nur Transkriptionsergebnis
            full_text = pipeline_result.transcription_result.get_full_text() if pipeline_result.transcription_result else "No transcription available"


        processing_time = time.time() - start_time

        result_obj = TranscriptionResult(
            job_id=job_id,
            file_name=original_file_name,
            municipality=municipality,
            meeting_date=meeting_date,
            meeting_type=meeting_type,
            transcription_text=full_text,
            processing_time=round(processing_time, 2),
            download_url=download_url or "",
            pipeline_statistics=pipeline_result.get_statistics(),
            quality_metrics={
                "pipeline_mode": config.mode.value,
                "preprocessing_mode": preprocessing_mode,
                "services_used": {
                    "transcription": True,
                    "alignment": pipeline_result.alignment_result is not None,
                    "diarization": pipeline_result.diarization_result is not None,
                    "protocol_formatting": pipeline_result.protocol_result is not None
                }
            }
        )

        # Save JSON result
        def _save_result():
            result_path = settings.results_dir / f"{job_id}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_obj.model_dump(), f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_save_result)

        # === STEP 5: COMPLETE JOB ===
        update_job(
            job_id,
            status="completed",
            progress_stage="finished",
            progress_percent=100.0,
            completed_at=datetime.now().isoformat(),
            download_url=download_url
        )

        logger.info(
            f"✅ Job {job_id} completed successfully in {processing_time:.2f}s "
            f"(Pipeline: {config.mode.value}, Speed: {audio_metadata.duration_seconds/processing_time:.1f}x realtime)"
        )

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}", exc_info=True)

        # Update job with error status
        job_info = get_job(job_id)
        current_progress = job_info.progress_percent if job_info else 0.0

        update_job(
            job_id,
            status="failed",
            progress_stage="error",
            progress_percent=current_progress,
            completed_at=datetime.now().isoformat()
        )

        # Save error details
        def _save_error():
            error_path = settings.results_dir / f"{job_id}_error.txt"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error in job {job_id} ({original_file_name}):\n")
                f.write(f"Pipeline mode: {pipeline_mode}\n")
                f.write(f"Preprocessing: {preprocessing_mode}\n")
                f.write(f"Stage: {job_info.progress_stage if job_info else 'unknown'}\n")
                f.write(f"Error: {str(e)}\n\n")
                f.write("Traceback:\n")
                import traceback
                traceback.print_exc(file=f)

        try:
            await asyncio.to_thread(_save_error)
            logger.info(f"🐛 Error details saved for job {job_id}")
        except Exception as save_error:
            logger.error(f"❌ Failed to save error details: {save_error}")

    finally:
        # Clean up upload file
        try:
            if file_path.exists():
                await asyncio.to_thread(file_path.unlink, True) # missing_ok=True für Python 3.8+
                logger.info(f"🗑️ Upload file cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"⚠️ Upload cleanup failed: {e}")

# ===== ENHANCED API ENDPOINTS =====
@app.get("/jobs/{job_id}", response_model=TranscriptionJob, tags=["Jobs"])
async def get_job_status(job_id: str):
    """Get job status with enhanced information"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job

@app.get("/jobs/", response_model=List[TranscriptionJob], tags=["Jobs"])
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List jobs with optional filtering"""
    all_jobs = list_all_jobs()

    if status:
        all_jobs = [job for job in all_jobs if job.status == status]

    # Sort by creation time (newest first) and limit
    sorted_jobs = sorted(all_jobs, key=lambda x: x.created_at, reverse=True)
    return sorted_jobs[:limit]

@app.get("/results/{job_id}", response_model=TranscriptionResult, tags=["Results"])
async def get_result(job_id: str):
    """Get transcription result with enhanced statistics"""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if job.status != "completed":
        raise HTTPException(400, f"Job {job_id} not completed (Status: {job.status})")

    result_path = settings.results_dir / f"{job_id}.json"
    if not result_path.exists():
        error_path = settings.results_dir / f"{job_id}_error.txt"
        if error_path.exists():
            with open(error_path, "r", encoding="utf-8") as f:
                error_details = f.read(500) # Read first 500 chars
            raise HTTPException(500, f"Job failed: {error_details}...")
        raise HTTPException(404, "Result file not found")

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return TranscriptionResult(**data)

@app.get("/pipeline/modes", tags=["Pipeline"])
async def get_pipeline_modes():
    """Get available pipeline modes with descriptions"""
    return {
        "modes": {
            "fast": {
                "description": "Quick processing with basic quality",
                "features": ["Basic transcription", "Optional alignment", "Fast speaker detection"],
                "typical_speed": "0.3x realtime", # Beispielwert
                "recommended_for": "Testing, quick previews, long recordings"
            },
            "balanced": {
                "description": "Good balance of quality and speed",
                "features": ["High-quality transcription", "Precise alignment", "Balanced diarization"],
                "typical_speed": "0.5x realtime", # Beispielwert
                "recommended_for": "Most use cases, production workflows"
            },
            "accurate": {
                "description": "High quality with detailed processing",
                "features": ["Premium transcription", "Precise alignment", "Accurate speaker detection"],
                "typical_speed": "0.8x realtime", # Beispielwert
                "recommended_for": "Important meetings, legal documents"
            },
            "premium": {
                "description": "Best possible quality (GPU required)",
                "features": ["Best transcription quality", "Phoneme-level alignment", "Advanced diarization"],
                "typical_speed": "1.2x realtime", # Beispielwert
                "recommended_for": "Critical recordings, archival purposes"
            }
        },
        "preprocessing_options": {
            "speech": {
                "description": "Optimized for speech (recommended)",
                "features": ["Noise reduction", "High-pass filtering", "RMS normalization", "Silence trimming"]
            },
            "minimal": {
                "description": "Minimal processing for high-quality audio",
                "features": ["Peak normalization only"]
            },
            "off": {
                "description": "No preprocessing (raw audio)",
                "features": ["Direct processing without modification"]
            }
        }
    }

@app.delete("/jobs/{job_id}", tags=["Jobs"])
async def delete_job_endpoint(job_id: str): # Name geändert, um Konflikt mit Variable zu vermeiden
    """Delete job and associated files with enhanced cleanup"""
    try:
        uuid.UUID(job_id) # Validate job ID format
    except ValueError:
        raise HTTPException(400, "Invalid job ID format")

    job_info = get_job(job_id)
    deleted_files = []

    async def _delete_file(path: Path):
        if path.exists():
            try:
                await asyncio.to_thread(path.unlink)
                deleted_files.append(path.name)
                logger.info(f"🗑️ Deleted file: {path.name}")
            except Exception as e:
                logger.warning(f"⚠️ Delete error {path.name}: {e}")

    # Delete all associated files
    await _delete_file(settings.results_dir / f"{job_id}.json")
    await _delete_file(settings.results_dir / f"{job_id}_error.txt")

    # Delete protocol file
    if job_info and job_info.download_url:
        protocol_filename = Path(job_info.download_url).name
        await _delete_file(settings.protocols_dir / protocol_filename)

    # Remove job from memory
    removed_job = remove_job(job_id)

    if not removed_job and not deleted_files:
         raise HTTPException(status_code=404, detail=f"Job {job_id} not found, no files to delete.")

    return {
        "message": f"Job {job_id} and associated files attempt to delete completed.",
        "deleted_files": deleted_files,
        "job_removed_from_memory": bool(removed_job)
    }

@app.get("/statistics", tags=["Service Info"])
async def get_service_statistics():
    """Get comprehensive service statistics"""
    all_jobs = list_all_jobs()

    # Job statistics
    job_stats = {
        "total_jobs": len(all_jobs),
        "by_status": {},
        "by_pipeline_mode": {},
        "processing_times": [] # Könnte noch implementiert werden, um Zeiten von abgeschlossenen Jobs zu sammeln
    }

    for job_instance in all_jobs: # Variable job umbenannt, um Konflikt zu vermeiden
        # Status distribution
        status = job_instance.status
        job_stats["by_status"][status] = job_stats["by_status"].get(status, 0) + 1

        # Pipeline mode distribution
        if job_instance.pipeline_mode:
            mode = job_instance.pipeline_mode
            job_stats["by_pipeline_mode"][mode] = job_stats["by_pipeline_mode"].get(mode, 0) + 1
        
        # Example for processing times (if stored in job or result)
        # if job_instance.status == "completed" and job_instance.completed_at and job_instance.created_at:
        #     # This would require parsing ISO format strings to datetime objects
        #     # processing_duration = (datetime.fromisoformat(job_instance.completed_at) - datetime.fromisoformat(job_instance.created_at)).total_seconds()
        #     # job_stats["processing_times"].append(processing_duration)
        #     pass


    # Hardware statistics
    hardware_stats = {
        "device": hardware_caps.device,
        "gpu_name": hardware_caps.gpu_name,
        "gpu_memory_gb": hardware_caps.gpu_memory_gb,
        "max_concurrent_jobs": hardware_caps.max_concurrent_jobs,
        "max_batch_size": hardware_caps.max_batch_size
    }

    # Service statistics
    service_stats = {
        "audio_service": {
            "supported_formats": len(audio_service.get_supported_formats()),
            "max_file_size_mb": settings.max_file_size / (1024 * 1024),
            "max_duration_hours": settings.max_audio_duration / 3600
        },
        "ml_pipeline": {
            "diarization_available": ml_pipeline.diarization_service.is_available,
            # default_model_loader ist hier nicht definiert, enhanced_model_loader verwenden
            "loaded_models": len(enhanced_model_loader.list_loaded_models()),
            "cache_info": ml_pipeline.alignment_service.get_cache_info()
        }
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "jobs": job_stats,
        "hardware": hardware_stats,
        "services": service_stats
    }

# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    import uvicorn

    # Sicherstellen, dass die Verzeichnisse existieren
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.protocols_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    Path("frontend").mkdir(parents=True, exist_ok=True) # Für Frontend-Dateien

    uvicorn.run(
        "app:app", # app:app sollte der Name der Datei sein, hier app.py, und app die FastAPI Instanz
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )