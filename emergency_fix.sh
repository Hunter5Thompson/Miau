#!/bin/bash
# emergency_fix.sh - Finde WAV2VEC2-Dateien und repariere sie ENDGÜLTIG

set -e

echo "🚨 NOTFALL-REPARATUR: WAV2VEC2-Dateien finden und kopieren"
echo "=========================================================="

TARGET_DIR="./model_cache/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-german"
SNAPSHOTS_MAIN="$TARGET_DIR/snapshots/main"

# 1. Container stoppen
echo "🛑 Stoppe Container..."
docker-compose stop protokoll-service-gpu

# 2. Finde Source-Verzeichnis
echo ""
echo "🔍 Suche nach WAV2VEC2-Source-Verzeichnis..."

SOURCE_CANDIDATES=(
    "./models--jonatasgrosman--wav2vec2-large-xlsr-53-german"
    "./cudnn-linux-x86_64-8.9.7.29_cuda12-archive/models--jonatasgrosman--wav2vec2-large-xlsr-53-german"
    "./*wav2vec2*german*"
    "./*/models--jonatasgrosman--wav2vec2-large-xlsr-53-german"
)

SOURCE_DIR=""
for candidate in "${SOURCE_CANDIDATES[@]}"; do
    if [ -d "$candidate" ]; then
        SOURCE_DIR="$candidate"
        echo "✅ Source gefunden: $SOURCE_DIR"
        break
    fi
done

# Fallback: Globale Suche
if [ -z "$SOURCE_DIR" ]; then
    echo "🔍 Globale Suche..."
    FOUND_DIRS=$(find . -name "*wav2vec2-large-xlsr-53-german*" -type d 2>/dev/null | head -5)
    if [ -n "$FOUND_DIRS" ]; then
        echo "📋 Gefundene Verzeichnisse:"
        echo "$FOUND_DIRS"
        SOURCE_DIR=$(echo "$FOUND_DIRS" | head -1)
        echo "✅ Verwende: $SOURCE_DIR"
    fi
fi

# Fallback: Suche nach einzelnen Dateien
if [ -z "$SOURCE_DIR" ]; then
    echo "🔍 Suche nach einzelnen WAV2VEC2-Dateien..."
    CONFIG_FILE=$(find . -name "config.json" | xargs grep -l "wav2vec2-large-xlsr-53" 2>/dev/null | head -1)
    if [ -n "$CONFIG_FILE" ]; then
        SOURCE_DIR=$(dirname "$CONFIG_FILE")
        echo "✅ Source durch config.json gefunden: $SOURCE_DIR"
    fi
fi

if [ -z "$SOURCE_DIR" ]; then
    echo "❌ KEIN WAV2VEC2-SOURCE GEFUNDEN!"
    echo ""
    echo "🔍 Manuelle Suche erforderlich:"
    echo "Verfügbare Verzeichnisse:"
    ls -la | grep -E "(wav2vec|jonatas|models--)"
    echo ""
    echo "📁 Alle Verzeichnisse mit 'models':"
    find . -name "*models*" -type d | head -10
    exit 1
fi

# 3. Validiere Source
echo ""
echo "🔍 Validiere Source: $SOURCE_DIR"
if [ ! -f "$SOURCE_DIR/config.json" ]; then
    echo "❌ config.json nicht gefunden in $SOURCE_DIR"
    echo "📋 Inhalt von $SOURCE_DIR:"
    ls -la "$SOURCE_DIR"
    exit 1
fi

if [ ! -f "$SOURCE_DIR/pytorch_model.bin" ]; then
    echo "❌ pytorch_model.bin nicht gefunden in $SOURCE_DIR"
    echo "🔍 Suche nach .bin Dateien:"
    find "$SOURCE_DIR" -name "*.bin" -o -name "*.safetensors"
    exit 1
fi

echo "✅ Source validiert: $SOURCE_DIR"

# 4. Erstelle Target-Struktur
echo ""
echo "🏗️ Erstelle Target-Struktur..."
mkdir -p "$SNAPSHOTS_MAIN"
mkdir -p "$TARGET_DIR/refs"
mkdir -p "$TARGET_DIR/blobs"

# 5. Kopiere ALLE Dateien
echo ""
echo "📋 Kopiere ALLE Dateien von $SOURCE_DIR nach $SNAPSHOTS_MAIN..."
cp -v "$SOURCE_DIR"/* "$SNAPSHOTS_MAIN/"

# 6. Erstelle refs/main
echo "main" > "$TARGET_DIR/refs/main"

# 7. Lösche alte Symlinks und erstelle neue
echo ""
echo "🔗 Erstelle neue Symlinks..."
cd "$TARGET_DIR"

# Lösche alte Symlinks
rm -f config.json pytorch_model.bin vocab.json special_tokens_map.json preprocessor_config.json

# Erstelle neue Symlinks für alle wichtigen Dateien
for file in config.json pytorch_model.bin vocab.json special_tokens_map.json preprocessor_config.json; do
    if [ -f "snapshots/main/$file" ]; then
        ln -sf "snapshots/main/$file" "$file"
        echo "  ✅ Link erstellt: $file"
    fi
done

cd - > /dev/null

# 8. Berechtigungen
echo ""
echo "🔐 Setze Berechtigungen..."
chmod -R 755 "$TARGET_DIR"

# 9. Verifikation
echo ""
echo "✅ FINALE VERIFIKATION:"
echo "📁 Target-Dir:"
ls -la "$TARGET_DIR"

echo ""
echo "📁 snapshots/main/:"
ls -la "$SNAPSHOTS_MAIN"

echo ""
echo "🔍 Kritische Dateien-Check:"
for file in config.json pytorch_model.bin; do
    if [ -f "$TARGET_DIR/$file" ]; then
        echo "  $file: ✅ Link funktioniert"
    else
        echo "  $file: ❌ FEHLT IMMER NOCH!"
    fi
done

# 10. Container starten
echo ""
echo "🚀 Starte Container..."
docker-compose start protokoll-service-gpu

echo ""
echo "🎉 NOTFALL-REPARATUR ABGESCHLOSSEN!"
echo "🧪 Teste jetzt eine Transkription mit Alignment!"

# 11. Quick-Test
echo ""
echo "⏳ Warte 10s und teste Container..."
sleep 10

docker-compose exec -T protokoll-service-gpu bash -c "
echo 'Container-Test:'
echo '  WAV2VEC2 config.json: \$([ -f /app/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-german/config.json ] && echo \"✅ GEFUNDEN\" || echo \"❌ FEHLT\")'
echo '  WAV2VEC2 pytorch_model.bin: \$([ -f /app/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-german/pytorch_model.bin ] && echo \"✅ GEFUNDEN\" || echo \"❌ FEHLT\")'
echo '  snapshots/main files: \$(ls -1 /app/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-german/snapshots/main/ 2>/dev/null | wc -l) Dateien'
"