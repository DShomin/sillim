python -m imet.main train model_1 \
--model seresnext101 \
--n-epochs 40 \
--batch-size 48 \
--size 288 \
--loss COMBINE \
--train_augments "random_crop, horizontal_flip, vertical_flip, random_rotate, color_jitter" \
--test_augments "random_crop, horizontal_flip, vertical_flip, random_rotate, color_jitter" \
--augment_ratio 0.5