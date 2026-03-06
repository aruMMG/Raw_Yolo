#!/usr/bin/env bash

# Images
for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/*; do
    ln -s "$f" ./datasets/parallel_data/images/train/
done

for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/isp/val/*; do
    ln -s "$f" ./datasets/parallel_data/images/val/
done

# Labels
for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/txt/train/*; do
    ln -s "$f" ./datasets/parallel_data/labels/train/
done

for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/txt/val/*; do
    ln -s "$f" ./datasets/parallel_data/labels/val/
done

# Raw
for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/*; do
    ln -s "$f" ./datasets/parallel_data/raw/train/
done

for f in /cifs/Shares/Raw_Bayer_Datasets/ROD/raw/val/*; do
    ln -s "$f" ./datasets/parallel_data/raw/val/
done
