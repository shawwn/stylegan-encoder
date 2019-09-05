#!/bin/bash
set -ex
mkdir -p datasets

if [ ! -d datasets/raw/animefaces ]
then
  mkdir -p datasets/raw
  python3 imguralbum.py https://imgur.com/a/zSzwjAY datasets/raw/animefaces
fi

if [ ! -d datasets/aligned/animefaces ]
then
  mkdir -p datasets/aligned/animefaces
  echo 'converting...'

  conv() {
    name="${1%.*}"
    echo converting "datasets/raw/animefaces/${name}.png" to 512x512 "datasets/aligned/animefaces/${name}.jpg"
    python3 convert_img.py "datasets/raw/animefaces/${name}.png" "datasets/aligned/animefaces/${name}.jpg"
  }
  export -f conv

  ls -1 datasets/raw/animefaces | xargs -n 1 -I {} bash -c 'conv "$@"' _ {}
fi

echo 'generating datasets/animefaces...'
python3 dataset_tool.py create_from_images datasets/animefaces datasets/aligned/animefaces
