mkdir -p data/raw
cd data/raw
for s in train val test-dev; do
  curl -L -o VisDrone2019-DET-$s.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-$s.zip
  unzip -q -o VisDrone2019-DET-$s.zip
done
