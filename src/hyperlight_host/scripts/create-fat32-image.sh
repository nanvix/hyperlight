#!/bin/bash

set -e

# Default values
SIZE="32M"
IMAGE_NAME="fat32.img"
NUM_DIRS=5
NUM_FILES=20
MIN_FILE_SIZE=32768        # 32KB minimum
MAX_FILE_SIZE=5242880      # 5MB maximum

# Help message
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Create a FAT32 image with randomly generated files and directories.

OPTIONS:
    -s, --size SIZE         Image size (default: 32M)
    -n, --name NAME         Output image filename (default: fat32.img)
    -d, --num-dirs N        Number of directories to create (default: 5)
    -f, --num-files N       Number of files to create (default: 20)
    --min-size BYTES        Minimum file size in bytes (default: 32768 / 32KB)
    --max-size BYTES        Maximum file size in bytes (default: 5242880 / 5MB)
    -h, --help              Show this help message

EXAMPLES:
    $(basename "$0") --size 64M --name test.img --num-dirs 10 --num-files 50
    $(basename "$0") -s 16M -d 3 -f 10 --min-size 50 --max-size 5000
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)
            SIZE="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -d|--num-dirs)
            NUM_DIRS="$2"
            shift 2
            ;;
        -f|--num-files)
            NUM_FILES="$2"
            shift 2
            ;;
        --min-size)
            MIN_FILE_SIZE="$2"
            shift 2
            ;;
        --max-size)
            MAX_FILE_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Word lists for generating random names
ADJECTIVES=("quick" "lazy" "happy" "sad" "bright" "dark" "old" "new" "big" "small" 
            "red" "blue" "green" "yellow" "purple" "orange" "warm" "cold" "soft" "hard"
            "fast" "slow" "loud" "quiet" "sharp" "smooth" "rough" "clean" "dirty" "fresh")

NOUNS=("fox" "dog" "cat" "bird" "fish" "tree" "rock" "cloud" "star" "moon"
       "sun" "river" "mountain" "forest" "desert" "ocean" "island" "castle" "bridge" "tower"
       "book" "lamp" "chair" "table" "door" "window" "garden" "flower" "leaf" "root")

EXTENSIONS=("txt" "dat" "bin" "log" "cfg" "xml" "json" "csv" "md" "doc")

# Generate a random name
generate_name() {
    local adj_idx=$((RANDOM % ${#ADJECTIVES[@]}))
    local noun_idx=$((RANDOM % ${#NOUNS[@]}))
    local num=$((RANDOM % 1000))
    echo "${ADJECTIVES[$adj_idx]}_${NOUNS[$noun_idx]}_${num}"
}

# Generate random content - pure random bytes
generate_content() {
    local size=$1
    dd if=/dev/urandom bs=4096 count=$(( (size + 4095) / 4096 )) 2>/dev/null | head -c "$size"
}

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "============================================"
echo "FAT32 Image Creator"
echo "============================================"
echo "Image size: $SIZE"
echo "Image name: $IMAGE_NAME"
echo "Directories: $NUM_DIRS"
echo "Files: $NUM_FILES"
echo "File size range: $MIN_FILE_SIZE - $MAX_FILE_SIZE bytes"
echo "Temp directory: $TEMP_DIR"
echo "============================================"
echo ""

# Create directories
declare -a DIRS
DIRS+=("")  # Root directory

echo "Creating directories..."
for ((i=0; i<NUM_DIRS; i++)); do
    dir_name=$(generate_name)
    mkdir -p "$TEMP_DIR/$dir_name"
    DIRS+=("$dir_name")
    echo "  [DIR]  /$dir_name"
done
echo ""

# Create files and distribute them randomly into directories
echo "Creating files..."
declare -a CREATED_FILES
for ((i=0; i<NUM_FILES; i++)); do
    file_name=$(generate_name)
    ext_idx=$((RANDOM % ${#EXTENSIONS[@]}))
    ext="${EXTENSIONS[$ext_idx]}"
    
    # Pick a random directory
    dir_idx=$((RANDOM % ${#DIRS[@]}))
    target_dir="${DIRS[$dir_idx]}"
    
    # Calculate random file size
    size_range=$((MAX_FILE_SIZE - MIN_FILE_SIZE))
    file_size=$((MIN_FILE_SIZE + RANDOM % (size_range + 1)))
    
    # Create the file
    if [[ -z "$target_dir" ]]; then
        file_path="$TEMP_DIR/${file_name}.${ext}"
        display_path="/${file_name}.${ext}"
    else
        file_path="$TEMP_DIR/$target_dir/${file_name}.${ext}"
        display_path="/$target_dir/${file_name}.${ext}"
    fi
    
    generate_content "$file_size" > "$file_path"
    actual_size=$(stat -c%s "$file_path")
    
    echo "  [FILE] $display_path ($actual_size bytes)"
    CREATED_FILES+=("$display_path:$actual_size")
done
echo ""

# Create the FAT32 image
echo "Creating FAT32 image: $IMAGE_NAME"

# Remove existing image if present
rm -f "$IMAGE_NAME"

# Create empty file of specified size
truncate -s "$SIZE" "$IMAGE_NAME"

# Format as FAT32
mkfs.fat -F 32 -n "RUSTFAT32" "$IMAGE_NAME" > /dev/null

echo "Copying files to image..."

# Copy directories first (mtools needs them to exist)
for dir in "${DIRS[@]}"; do
    if [[ -n "$dir" ]]; then
        mmd -i "$IMAGE_NAME" "::/$dir" 2>/dev/null || true
    fi
done

# Copy all files
for dir in "${DIRS[@]}"; do
    if [[ -z "$dir" ]]; then
        # Root directory files
        for file in "$TEMP_DIR"/*; do
            if [[ -f "$file" ]]; then
                mcopy -i "$IMAGE_NAME" "$file" "::" 2>/dev/null || true
            fi
        done
    else
        # Subdirectory files
        for file in "$TEMP_DIR/$dir"/*; do
            if [[ -f "$file" ]]; then
                mcopy -i "$IMAGE_NAME" "$file" "::/$dir/" 2>/dev/null || true
            fi
        done
    fi
done

echo ""
echo "============================================"
echo "Image created successfully: $IMAGE_NAME"
echo "============================================"
echo ""

# List contents of the created image
echo "Contents of $IMAGE_NAME:"
echo "--------------------------------------------"
mdir -i "$IMAGE_NAME" -/ -a ::
echo "--------------------------------------------"
echo ""
echo "Done!"
