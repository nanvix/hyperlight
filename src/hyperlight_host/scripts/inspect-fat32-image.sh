#!/bin/bash

set -e

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") IMAGE_FILE

Inspect a FAT32 image and list its contents, directories, and metadata.

ARGUMENTS:
    IMAGE_FILE      Path to the FAT32 image file

OPTIONS:
    -h, --help      Show this help message

EXAMPLES:
    $(basename "$0") fat32.img
    $(basename "$0") /path/to/my_image.img
EOF
}

# Check arguments
if [[ $# -eq 0 ]]; then
    echo "Error: No image file specified"
    show_help
    exit 1
fi

case $1 in
    -h|--help)
        show_help
        exit 0
        ;;
esac

IMAGE_FILE="$1"

# Check if file exists
if [[ ! -f "$IMAGE_FILE" ]]; then
    echo "Error: File not found: $IMAGE_FILE"
    exit 1
fi

echo "============================================"
echo "FAT32 Image Inspector"
echo "============================================"
echo "Image: $IMAGE_FILE"
echo "Size: $(stat -c%s "$IMAGE_FILE") bytes ($(du -h "$IMAGE_FILE" | cut -f1))"
echo "============================================"
echo ""

# Volume information
echo ">> Volume Information:"
echo "--------------------------------------------"
minfo -i "$IMAGE_FILE" :: 2>/dev/null || echo "Could not read volume info"
echo "--------------------------------------------"
echo ""

# Boot sector info
echo ">> Boot Sector Details:"
echo "--------------------------------------------"
minfo -i "$IMAGE_FILE" -v :: 2>/dev/null | head -50 || echo "Could not read boot sector"
echo "--------------------------------------------"
echo ""

# Directory listing (recursive)
echo ">> Directory Listing (recursive):"
echo "--------------------------------------------"
mdir -i "$IMAGE_FILE" -/ -a :: 2>/dev/null || echo "Could not list directories"
echo "--------------------------------------------"
echo ""

# File count and size statistics
echo ">> Statistics:"
echo "--------------------------------------------"

# Count files and directories
file_count=$(mdir -i "$IMAGE_FILE" -/ -a :: 2>/dev/null | grep -v "^ " | grep -v "^$" | grep -v "Volume" | grep -v "Directory" | grep -v "total" | grep -v "<DIR>" | wc -l || echo "0")
dir_count=$(mdir -i "$IMAGE_FILE" -/ -a :: 2>/dev/null | grep "<DIR>" | wc -l || echo "0")

# Get total size from mdir output
total_line=$(mdir -i "$IMAGE_FILE" :: 2>/dev/null | grep "bytes free" || echo "")

echo "Files: ~$file_count"
echo "Directories: ~$dir_count"
if [[ -n "$total_line" ]]; then
    echo "Space info: $total_line"
fi
echo "--------------------------------------------"
echo ""

# List all files with full paths
echo ">> All Files (full paths):"
echo "--------------------------------------------"
list_files_recursive() {
    local path="$1"
    local indent="$2"
    
    mdir -i "$IMAGE_FILE" -a "::$path" 2>/dev/null | while read -r line; do
        # Skip header lines
        if [[ "$line" == *"Volume"* ]] || [[ "$line" == *"Directory"* ]] || [[ -z "$line" ]] || [[ "$line" == " "* ]] || [[ "$line" == *"bytes"* ]] || [[ "$line" == *"file"* ]]; then
            continue
        fi
        
        # Parse the line - mdir format varies, try to extract name
        name=$(echo "$line" | awk '{print $1}')
        
        # Skip . and ..
        if [[ "$name" == "." ]] || [[ "$name" == ".." ]]; then
            continue
        fi
        
        if [[ "$line" == *"<DIR>"* ]]; then
            echo "${indent}[DIR] $path/$name"
            list_files_recursive "$path/$name" "  $indent"
        else
            # Extract size if possible
            size=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/ && $i > 0) print $i}' | head -1)
            if [[ -n "$size" ]]; then
                echo "${indent}[FILE] $path/$name ($size bytes)"
            else
                echo "${indent}[FILE] $path/$name"
            fi
        fi
    done
}

# Simple recursive listing using mdir -/
mdir -i "$IMAGE_FILE" -/ :: 2>/dev/null | grep -v "^$" | grep -v "Volume" | grep -v "Directory" | grep -v "bytes" | grep -v "files" | head -100 || echo "No files found"

echo "--------------------------------------------"
echo ""
echo "Done!"
