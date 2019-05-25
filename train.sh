<<<<<<< HEAD
python -m imet.main train model_1 \
--model resnet50 \
--n-epochs 40 \
--batch-size 1 \
--size 288 \
--loss COMBINE \
--limit 1 \
--train_augments "random_crop, color_jitter" \
--test_augments "random_crop, color_jitter" \
--augment_ratio 1.0
||||||| merged common ancestors
python -m imet.main train model_1 \
--model resnet50 \
--n-epochs 40 \
--batch-size 1 \
--size 288 \
--loss COMBINE \
--train_augments "random_crop, color_jitter" \
--test_augments "random_crop, color_jitter" \
--augment_ratio 1.0
=======
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
>>>>>>> develop
