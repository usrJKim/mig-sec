# 구조: ./data/n0172345/images/*.JPEG → ./data/n0172345/*.JPEG
for d in ./data/*; do
  if [ -d "$d/images" ]; then
    mv "$d/images"/* "$d"/
    rmdir "$d/images"
  fi
done

