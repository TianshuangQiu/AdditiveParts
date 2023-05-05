find ../../scripts/simplified/output -name "*.obj" -type f -exec sh -c '
   ./manifold "$0" "$0" 500
   ' {} \;

find ../../scripts/simplified/output -name "*.obj" -type f -exec sh -c '
   ./simplify -i "$0" -o "$0" -m -f 500
   ' {} \;

