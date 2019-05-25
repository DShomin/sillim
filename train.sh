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
