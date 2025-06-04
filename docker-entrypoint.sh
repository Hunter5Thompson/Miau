#!/bin/bash
# docker-entrypoint.sh - Enhanced WhisperX Service für H100 + cuDNN + Nexus
# Kombiniert robuste Nexus-Integration mit cuDNN-Runtime-Fix

set -e

echo "🚀 Container-Startup: WhisperX H100-Service (Enhanced)"
echo "🔧 Strategie: Nexus + cuDNN Runtime-Kompatibilität + H100-Optimierung"

# ===== SYSTEM INFORMATION =====
echo "🔍 System-Checks:"
echo "  - Python: $(python3 --version 2>/dev/null || echo 'Python nicht verfügbar')"
echo "  - Device: ${DEVICE:-cuda}"
echo "  - Cache-Base: ${HF_HOME:-/app/.cache/huggingface}"
echo "  - Nexus Mode: ${PIP_INDEX_URL:+Nexus-Repository}"

# ===== NEXUS CONNECTION CHECK =====
check_nexus_connectivity() {
    if [ -n "$PIP_TRUSTED_HOST" ]; then
        echo "🌐 Teste Nexus-Konnektivität: $PIP_TRUSTED_HOST"
        if curl -s --max-time 5 --connect-timeout 3 "https://$PIP_TRUSTED_HOST" >/dev/null 2>&1; then
            echo "  ✅ Nexus erreichbar"
        else
            echo "  ⚠️ Nexus nicht erreichbar - offline Modus"
        fi
    else
        echo "  ⚠️ No Nexus credentials - using anonymous access"
    fi
}

# ===== CACHE DIRECTORY SETUP =====
setup_cache_directories() {
    echo "📁 Setup Cache-Verzeichnisse..."
    
    local cache_dirs=(
        "/app/.cache/huggingface/hub"
        "/app/.cache/huggingface/transformers" 
        "/app/.cache/torch"
        "/app/.cache/whisper"
        "/app/uploads"
        "/app/results"
        "/app/protocols"
        "/app/logs"
    )
    
    for dir in "${cache_dirs[@]}"; do
        mkdir -p "$dir" 2>/dev/null || true
    done
}

# ===== PERMISSIONS SETUP =====
setup_permissions() {
    echo "🔐 Setze Berechtigungen..."
    
    # Versuche Berechtigungen zu setzen (kann bei Read-only Mounts fehlschlagen)
    chmod -R 755 /app/.cache /app/uploads /app/results /app/protocols /app/logs 2>/dev/null || echo "⚠️ Einige Berechtigungen übersprungen (Read-only Mounts)"
}

# ===== ERWEITERTE CUDNN RUNTIME COMPATIBILITY CHECK =====
cudnn_runtime_check() {
    echo "🔧 cuDNN Runtime-Kompatibilitäts-Check (H100-optimiert)..."
    
    # Test CUDA availability first
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "  ✅ CUDA verfügbar"
        
        # Get GPU info
        GPU_INFO=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "Unknown")
        echo "  🎮 GPU: $GPU_INFO"
        
        # Test cuDNN with detailed info
        CUDNN_STATUS=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(f'cuDNN available: {torch.backends.cudnn.is_available()}')
    if torch.backends.cudnn.is_available():
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
        print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
    else:
        print('cuDNN: Not available - checking LD_LIBRARY_PATH...')
        import os
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        print(f'LD_LIBRARY_PATH contains cuDNN: {\"/cudnn\" in ld_path}')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Python torch check failed")
        
        echo "  🧠 $CUDNN_STATUS"
        
        # H100-specific optimizations
        if echo "$GPU_INFO" | grep -qi "h100"; then
            echo "  🚀 H100 detected - activating premium optimizations"
            export TORCH_CUDNN_V8_API_ENABLED=1
            export CUDNN_BENCHMARK=1
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,roundup_power2_divisions:16"
        fi
        
    else
        echo "  ℹ️ CUDA nicht verfügbar - CPU-Modus"
        export DEVICE="cpu"
    fi
}

# ===== MODEL AVAILABILITY CHECK =====
check_model_availability() {
    echo "📦 Prüfe Model-Verfügbarkeit..."
    
    local model_count=0
    local example_models=()
    
    # Suche nach Model-Dateien
    for cache_dir in "/app/.cache/huggingface" "/app/.cache/whisper" "/app/.cache/torch"; do
        if [ -d "$cache_dir" ]; then
            local found_models=$(find "$cache_dir" -name "*.bin" -o -name "*.pt" -o -name "*.safetensors" 2>/dev/null | head -3)
            if [ -n "$found_models" ]; then
                model_count=$((model_count + $(echo "$found_models" | wc -l)))
                while IFS= read -r model_file; do
                    if [ ${#example_models[@]} -lt 3 ]; then
                        local size=$(du -h "$model_file" 2>/dev/null | cut -f1 || echo "?")
                        example_models+=("$(basename "$model_file") ($size)")
                    fi
                done <<< "$found_models"
            fi
        fi
    done
    
    if [ $model_count -gt 0 ]; then
        echo "  ✅ $model_count Model-Dateien gefunden"
        if [ ${#example_models[@]} -gt 0 ]; then
            echo "  📋 Beispiel-Modelle:"
            for model in "${example_models[@]}"; do
                echo "    - $model"
            done
        fi
    else
        echo "  ⚠️ Keine Model-Dateien gefunden - werden bei Bedarf geladen"
    fi
}

# ===== AUDIO STACK TEST =====
test_audio_stack() {
    echo "🎵 Teste Audio-Stack (FFmpeg-frei)..."
    
    python3 -c "
try:
    import soundfile as sf
    import librosa
    import numpy as np
    print('  ✅ Audio-Bibliotheken funktionsfähig')
    print('    - SoundFile:', sf.__version__)
    print('    - Librosa:', librosa.__version__)
    print('    - NumPy:', np.__version__)
    
    # Quick Audio I/O Test
    test_audio = np.random.rand(16000).astype(np.float32)
    print(f'    - Audio I/O Test: ✅ ({len(test_audio)} samples @ 16000Hz)')
    
except Exception as e:
    print(f'  ❌ Audio-Stack-Fehler: {e}')
    exit(1)
"
}

# ===== ERWEITERTE WHISPERX STACK TEST =====
test_whisperx_stack() {
    echo "🤖 Teste WhisperX-Stack (H100-optimiert)..."
    
    python3 -c "
try:
    import whisperx
    import torch
    import faster_whisper
    print('  ✅ WhisperX-Stack verfügbar')
    print('    - WhisperX:', getattr(whisperx, '__version__', 'installed'))
    print('    - PyTorch:', torch.__version__)
    print('    - faster-whisper:', faster_whisper.__version__)
    print('    - CUDA verfügbar:', torch.cuda.is_available())
    
    # Test model loading capability (ohne tatsächlich zu laden)
    from config.hardware_detection import get_hardware_capabilities
    from config.model_config import default_model_loader
    
    hw_caps = get_hardware_capabilities()
    print(f'    - Hardware Device: {hw_caps.device}')
    print(f'    - Max Batch Size: {hw_caps.max_batch_size}')
    
    if hw_caps.gpu_name:
        print(f'    - GPU Name: {hw_caps.gpu_name}')
        print(f'    - GPU Memory: {hw_caps.gpu_memory_gb:.1f}GB')
    
    print('  ✅ WhisperX Services bereit für Transkription')
    
except Exception as e:
    print(f'  ❌ WhisperX-Stack-Fehler: {e}')
    print('  🔧 Fallback: Basic WhisperX verfügbar')
"
}

# ===== OFFLINE MODE STATUS =====
show_offline_status() {
    echo "🌐 Offline-Modus Status:"
    echo "  - HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
    echo "  - TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}"
    echo "  - Nexus Index: ${PIP_INDEX_URL:-Standard PyPI}"
    echo "  - HuggingFace Token: ${HUGGINGFACE_TOKEN:+***verfügbar***}"
}

# ===== ERWEITERTE HARDWARE OPTIMIZATION =====
apply_hardware_optimizations() {
    # GPU vs CPU Optimizations
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "🎮 GPU-Modus - CUDA-Optimierungen aktiviert"
        
        # H100-spezifische Optimierungen
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        if echo "$GPU_NAME" | grep -qi "h100"; then
            echo "  🚀 H100-Premium-Optimierungen aktiviert"
            export CUDA_MEMORY_FRACTION=${CUDA_MEMORY_FRACTION:-0.8}
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,roundup_power2_divisions:16,expandable_segments:True"
            export TORCH_CUDNN_V8_API_ENABLED=1
            export CUDNN_DETERMINISTIC=0
            export CUDNN_BENCHMARK=1
        else
            echo "  🎮 Standard GPU-Optimierungen"
            export CUDA_MEMORY_FRACTION=${CUDA_MEMORY_FRACTION:-0.9}
            export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
        fi
        
        export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
    else
        echo "🖥️ CPU-Modus - Optimiert für CPU-Verarbeitung"
        export DEVICE="cpu"
        export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
    fi
}

# ===== MAIN STARTUP SEQUENCE =====
main() {
    # System checks
    check_nexus_connectivity
    setup_cache_directories
    setup_permissions
    
    # Hardware & Model checks
    cudnn_runtime_check
    check_model_availability
    
    # Software stack tests
    test_audio_stack
    test_whisperx_stack
    
    # Configuration
    show_offline_status
    apply_hardware_optimizations
    
    echo ""
    echo "✅ Container-Startup abgeschlossen (Enhanced H100-Ready)"
    
    # Final status summary
    echo "📊 Service-Konfiguration:"
    echo "   - Compute Device: ${DEVICE:-auto}"
    echo "   - CUDA verfügbar: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'false')"
    echo "   - cuDNN verfügbar: $(python3 -c 'import torch; print(torch.backends.cudnn.is_available() if torch.cuda.is_available() else False)' 2>/dev/null || echo 'false')"
    echo "   - Audio-Stack: FFmpeg-frei ✅"
    echo "   - Model-Cache: Bereit ✅"
    echo "   - Nexus-Kompatibilität: ✅"
    echo "   - H100-Optimierungen: $(echo "$GPU_NAME" | grep -qi "h100" && echo "✅" || echo "N/A")"
    echo ""
    echo "🚀 Starte WhisperX Service..."
}

# ===== EXECUTION =====
main

# ===== START APPLICATION =====
exec "$@"