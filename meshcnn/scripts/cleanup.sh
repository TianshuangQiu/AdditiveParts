#!/bin/bash

find ./sdata -name "*.obj" -type f -exec sh -c '
   /Applications/Blender.app/Contents/MacOS/Blender --background --python blenderClean.py "$0" 750 "$0"
   ' {} \;
