#!/bin/bash
set -e

CODEGEN="./target/release/embedded-bitmap-fonts-codegen"
BITMAP_DIR="./fonts/bitmap"
OUT_DIR="./src"

# Font families to generate
# Format: name[:include_pattern:exclude_pattern]
FONTS=(
    "artwiz"
    "bitocra"
    "cherry"
    "creep"
    "ctrld"
    "dina"
    "dylex"
    "envypn"
    "gohufont"
    "gomme"
    "haxor"
    "jmk"
    "kakwa"
    "knxt"
    "lokaltog"
    "mplus"
    "orp"
    "peep"
    "phallus"
    "progsole"
    "scientifica"
    "spleen"
    "tamzen:Tamzen:Powerline"
    "terminus"
    "tewi"
    "trisk"
    "xbmicons"
)

for entry in "${FONTS[@]}"; do
    IFS=':' read -r name pattern exclude <<< "$entry"
    
    echo "Generating $name..."
    
    args=(-i "$BITMAP_DIR/$name" -o "$OUT_DIR" -f "$name")
    [ -n "$pattern" ] && args+=(-P "$pattern")
    [ -n "$exclude" ] && args+=(-X "$exclude")
    
    if $CODEGEN "${args[@]}" 2>&1; then
        echo "  OK: $name"
    else
        echo "  FAILED: $name"
    fi
done

echo "Done!"
