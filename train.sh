python -m imet.main train seresnext101_320_bce_0 \
--model seresnext101 \
--n-epochs 30 \
--batch-size 16 \
--size 320 \
--loss COMBINE2 \
--train_augments "random_crop, horizontal_flip, vertical_flip, random_rotate, color_jitter" \
--test_augments "random_crop, horizontal_flip, vertical_flip" \
--augment_ratio 0.5 \
--worker 16