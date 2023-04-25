#!/bin/bash

search_and_process() {
   find ../data -name "*.stl" -type f | head -n 5000 | xargs -I {} sh -c '
   /Applications/Blender.app/Contents/MacOS/Blender --background --python stl_to_obj.py -- "$0"
   ' {}
   
   find ./output -name "*.obj" -type f | head -n 5000 | xargs -I {} sh -c '
   /Applications/Blender.app/Contents/MacOS/Blender --background --python normalize.py "$0" 512 "$0"
   ' {}
}
echo "beginning"
search_and_process

