#!/bin/bash
# usage: ./images/src/render.sh day01_vector_addition   -> images/day01_vector_addition.png
cd "$(dirname "$0")"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
  --force-device-scale-factor=2 --window-size=1600,900 --virtual-time-budget=8000 \
  --screenshot="$PWD/../$1.png" "file://$PWD/$1.html" 2>/dev/null
