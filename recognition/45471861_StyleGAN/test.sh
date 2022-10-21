DATA_DIR=./keras_png_slices_data/keras_png_slices_train/keras_png_slices_train
OUTPUT_DIR=./Output

python run.py --resolution 256 --epochs 40 --batch 32 --lr 0.0001 $DATA_DIR $OUTPUT_DIR 8 512 8 32 80
