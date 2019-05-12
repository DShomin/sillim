# Sillim Kaggle Repo

## IMet Hyper-prameters Information Google Link (Base Line)
 - https://docs.google.com/spreadsheets/d/1ixBPEpVJpNaL1PupwG3GmdPpoDNk5qP9V976io2__AQ/edit?usp=sharing
 
## Weight Model File Upload Link Google
 - https://drive.google.com/drive/folders/1MU8nPdU55zXmp3TmR-xwam57EVtyc-rC?usp=sharing

## References

- build script(https://github.com/lopuhin/kaggle-script-template)
- pytorch template(https://github.com/victoresque/pytorch-template)

## Added datasets

- resnet50(https://www.kaggle.com/pytorch/resnet50)
- se-resnext(https://www.kaggle.com/seefun/se-resnext-pytorch-pretrained)


# Arguments

- model(default: resnet50): 아래의 모델의 종류 중에서 선택합니다.
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152
    - densenet121
    - densenet169
    - densenet201
    - densenet161
    - seresnext50
    - seresnext101


- train_augments(default: "random_crop, horizontal_flip): 아래에서 원하는 것을 넣거나 빼고 string으로 값을 줍니다.
- test_augments(default: "random_crop, horizontal_flip): 아래에서 원하는 것을 넣거나 빼고 string으로 값을 줍니다.
    - "random_crop, keep_aspect, horizontal_flip, vertical_flip, random_rotate, color_jitter"
    - ex: trainig 시 random crop과 horizontal flip만 원할 시 --train_augments "random_crop, horizontal_flip"
    - ex: test 시 random crop과 horizontal flip만 원할 시 --test_augments "random_crop, horizontal_flip"


- size(default: 288): 입력 영상의 크기를 설정합니다.
- augment_ratio(default: 0.5): augmentation이 적용되는 확률입니다.

- loss(default: "BCE"): loss를 선택합니다.
    - "BCE": binary cross entropy를 사용합니다.
    - "FOCAL": focal loss를 사용합니다.
    - "FBET": Fbet loss를 사용합니다.
    - "COMBINE": focal과 fbet을 함께 사용합니다.
    - ex: --loss COMBINE