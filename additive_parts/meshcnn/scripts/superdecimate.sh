#!/bin/bash

search_and_process() {
   #find ../data -name "*.stl" -type f -exec sh -c '
   #/Applications/Blender.app/Contents/MacOS/Blender --background --python stl_to_obj.py -- "$0"
   #' {} \;

   find ./simplified -name "*.obj" -type f -exec sh -c '
   /Applications/Blender.app/Contents/MacOS/Blender --background --python decimate.py "$0" 750 "$0"
   ' {} \;
}
echo "beginning"
search_and_process

