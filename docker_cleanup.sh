#!/bin/bash
# docker_cleanup.sh - Findet und beseitigt Docker-Leichen

set -e

echo "🧟‍♂️ Docker-Leichen-Detektor gestartet..."
echo "=========================================="

# Funktion: Dateigröße human-readable
format_size() {
    local bytes=$1
    if [ $bytes -gt 1073741824 ]; then
        echo "$(( bytes / 1073741824 ))GB"
    elif [ $bytes -gt 1048576 ]; then
        echo "$(( bytes / 1048576 ))MB"
    else
        echo "$(( bytes / 1024 ))KB"
    fi
}

# Funktion: Docker-Speicherplatz-Analyse
analyze_docker_space() {
    echo "💾 Docker-Speicherplatz-Analyse:"
    echo "================================"
    
    # Docker system df (wie Linux df aber für Docker)
    if command -v docker &> /dev/null; then
        docker system df -v 2>/dev/null || docker system df
    else
        echo "❌ Docker nicht verfügbar"
        return 1
    fi
    
    echo ""
}

# 1. 🧟‍♂️ TOTE CONTAINER FINDEN
find_dead_containers() {
    echo "🧟‍♂️ Suche tote Container..."
    echo "============================"
    
    # Alle Container (auch gestoppte)
    TOTAL_CONTAINERS=$(docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | wc -l)
    echo "📊 Gesamt-Container: $((TOTAL_CONTAINERS - 1))"
    
    # Laufende Container
    RUNNING_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Status}}" | wc -l)
    echo "✅ Laufende Container: $((RUNNING_CONTAINERS - 1))"
    
    # Tote Container
    DEAD_CONTAINERS=$(docker ps -f "status=exited" --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | wc -l)
    if [ $DEAD_CONTAINERS -gt 1 ]; then
        echo "💀 Tote Container: $((DEAD_CONTAINERS - 1))"
        echo ""
        docker ps -f "status=exited" --format "table {{.Names}}\t{{.Status}}\t{{.Size}}"
    else
        echo "✅ Keine toten Container gefunden"
    fi
    
    # Verwaiste Container (created but never started)
    ORPHANED_CONTAINERS=$(docker ps -f "status=created" --format "table {{.Names}}\t{{.Status}}" | wc -l)
    if [ $ORPHANED_CONTAINERS -gt 1 ]; then
        echo ""
        echo "👻 Verwaiste Container: $((ORPHANED_CONTAINERS - 1))"
        docker ps -f "status=created" --format "table {{.Names}}\t{{.Status}}"
    fi
    
    echo ""
}

# 2. 🖼️ VERWAISTE IMAGES FINDEN
find_orphaned_images() {
    echo "🖼️ Suche verwaiste Images..."
    echo "============================="
    
    # Dangling Images (keine Tags, nicht verwendet)
    DANGLING_IMAGES=$(docker images -f "dangling=true" -q | wc -l)
    if [ $DANGLING_IMAGES -gt 0 ]; then
        echo "👻 Dangling Images: $DANGLING_IMAGES"
        docker images -f "dangling=true" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    else
        echo "✅ Keine dangling Images"
    fi
    
    # Unbenutzte Images
    echo ""
    echo "🔍 Images ohne aktive Container:"
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -20
    
    echo ""
}

# 3. 💾 VERWAISTE VOLUMES FINDEN  
find_orphaned_volumes() {
    echo "💾 Suche verwaiste Volumes..."
    echo "============================="
    
    # Dangling Volumes
    DANGLING_VOLUMES=$(docker volume ls -f "dangling=true" -q | wc -l)
    if [ $DANGLING_VOLUMES -gt 0 ]; then
        echo "👻 Verwaiste Volumes: $DANGLING_VOLUMES"
        docker volume ls -f "dangling=true"
        
        # Größe schätzen
        echo ""
        echo "📊 Volume-Größen:"
        docker volume ls -f "dangling=true" -q | while read vol; do
            if [ -n "$vol" ]; then
                vol_path=$(docker volume inspect "$vol" --format '{{.Mountpoint}}' 2>/dev/null || echo "unknown")
                if [ -d "$vol_path" ]; then
                    vol_size=$(du -sh "$vol_path" 2>/dev/null | cut -f1 || echo "?")
                    echo "  $vol: $vol_size"
                fi
            fi
        done
    else
        echo "✅ Keine verwaisten Volumes"
    fi
    
    echo ""
}

# 4. 🌐 VERWAISTE NETWORKS FINDEN
find_orphaned_networks() {
    echo "🌐 Suche verwaiste Networks..."
    echo "=============================="
    
    # Alle Networks
    ALL_NETWORKS=$(docker network ls --format "{{.Name}}" | grep -v -E "^(bridge|host|none)$" | wc -l)
    echo "📊 Custom Networks: $ALL_NETWORKS"
    
    if [ $ALL_NETWORKS -gt 0 ]; then
        echo "🔍 Custom Networks:"
        docker network ls --format "table {{.Name}}\t{{.Driver}}\t{{.CreatedAt}}" | grep -v -E "bridge.*ago|host.*ago|none.*ago"
    fi
    
    echo ""
}

# 5. 🗑️ BUILD-CACHE FINDEN
find_build_cache() {
    echo "🗑️ Docker Build-Cache..."
    echo "========================"
    
    # Build-Cache analysieren
    if docker builder du 2>/dev/null; then
        echo "📊 Build-Cache gefunden"
    else
        echo "ℹ️ Build-Cache-Info nicht verfügbar (Docker < 19.03)"
    fi
    
    echo ""
}

# 6. 🧹 CLEANUP-OPTIONEN
show_cleanup_options() {
    echo "🧹 CLEANUP-OPTIONEN:"
    echo "===================="
    echo ""
    echo "SICHERE CLEANUPS:"
    echo "  docker system prune              # Entfernt gestoppte Container, unbenutzte Networks, dangling Images"
    echo "  docker container prune           # Nur tote Container"
    echo "  docker image prune               # Nur dangling Images"
    echo "  docker volume prune              # Nur verwaiste Volumes"
    echo "  docker network prune             # Nur unbenutzte Networks"
    echo ""
    echo "AGGRESSIVE CLEANUPS:"
    echo "  docker system prune -a           # Alles + unbenutzte Images"
    echo "  docker system prune -a --volumes # Alles + Volumes (⚠️ VORSICHT!)"
    echo ""
    echo "MANUELLE CLEANUPS:"
    echo "  docker rm \$(docker ps -aq)      # Alle Container löschen"
    echo "  docker rmi \$(docker images -aq) # Alle Images löschen (⚠️ EXTREM VORSICHTIG!)"
    echo ""
}

# 7. 🎯 INTERAKTIVER CLEANUP
interactive_cleanup() {
    echo "🎯 Interaktiver Cleanup:"
    echo "========================"
    echo ""
    echo "Was möchtest du bereinigen?"
    echo "1) 💀 Nur tote Container"
    echo "2) 🖼️  Nur dangling Images" 
    echo "3) 💾 Nur verwaiste Volumes"
    echo "4) 🌐 Nur unbenutzte Networks"
    echo "5) 🧹 Alles sicher (prune ohne -a)"
    echo "6) 💥 Alles aggressiv (prune -a)"
    echo "7) 🚫 Nichts - nur anzeigen"
    echo ""
    
    read -p "Wähle Option (1-7): " choice
    
    case $choice in
        1)
            echo "💀 Entferne tote Container..."
            docker container prune -f
            ;;
        2)
            echo "🖼️ Entferne dangling Images..."
            docker image prune -f
            ;;
        3)
            echo "💾 Entferne verwaiste Volumes..."
            docker volume prune -f
            ;;
        4)
            echo "🌐 Entferne unbenutzte Networks..."
            docker network prune -f
            ;;
        5)
            echo "🧹 Sichere Bereinigung..."
            docker system prune -f
            ;;
        6)
            echo "💥 Aggressive Bereinigung..."
            echo "⚠️ WARNUNG: Entfernt auch unbenutzte Images!"
            read -p "Wirklich fortfahren? (y/N): " confirm
            if [[ $confirm =~ ^[Yy] ]]; then
                docker system prune -a -f
            else
                echo "Abgebrochen."
            fi
            ;;
        7)
            echo "👀 Nur Analyse - keine Bereinigung"
            ;;
        *)
            echo "❌ Ungültige Option"
            ;;
    esac
}

# 8. 📊 HAUPT-ANALYSE
main_analysis() {
    echo "🔍 Starte Docker-Leichen-Analyse..."
    echo ""
    
    # Speicherplatz vor Analyse
    analyze_docker_space
    
    # Verschiedene Leichen-Typen finden
    find_dead_containers
    find_orphaned_images  
    find_orphaned_volumes
    find_orphaned_networks
    find_build_cache
    
    echo "🎯 ZUSAMMENFASSUNG:"
    echo "=================="
    echo "Tote Container: $(docker ps -f 'status=exited' -q | wc -l)"
    echo "Dangling Images: $(docker images -f 'dangling=true' -q | wc -l)" 
    echo "Verwaiste Volumes: $(docker volume ls -f 'dangling=true' -q | wc -l)"
    echo "Custom Networks: $(docker network ls --format '{{.Name}}' | grep -v -E '^(bridge|host|none)$' | wc -l)"
    echo ""
}

# HAUPTPROGRAMM
main_analysis
show_cleanup_options

# Interaktiver Cleanup anbieten
echo ""
read -p "🧹 Interaktiven Cleanup starten? (y/N): " start_cleanup
if [[ $start_cleanup =~ ^[Yy] ]]; then
    interactive_cleanup
    echo ""
    echo "✅ Cleanup abgeschlossen!"
    echo ""
    echo "📊 Neuer Speicherplatz-Status:"
    analyze_docker_space
fi

echo ""
echo "🎉 Docker-Leichen-Analyse abgeschlossen!"