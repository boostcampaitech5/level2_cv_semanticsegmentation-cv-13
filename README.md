# CV_Sementic-Segmentation
Classification Pipeline in Computer Vision (Pytorch)

# Environments


# Directory

```bash
CV_classification
├── datasets
│   ├── __init__.py
│   ├── augmentation.py
│   └── factory.py
├── models
│   ├── __init__.py
│   ├── resnet.py
│   └── loss.py
├── log.py
├── main.py
├── train.py
├── run.sh
├── requirements.txt
├── README.md
└── LICENSE
```

# Pipeline

0. Set seed
1. Make directory to save results
2. Build model
3. Build dataset with augmentations
   - Train dataset
   - Validation dataset (optional)
   - Test dataset 
4. Make dataLoader
5. Define optimizer (model parameters)
6. Define loss function
7. Training model
   - Checkpoint model using evaluation on validation dataset
   - Log training history using `logging` or `wandb` in save folder
8. Testing model




**example**

```bash
bash run.sh CIFAR10 10
```

# Config Parameters
### Wandb 관련 설정
|Argument|Description|Default|Possible value|
|---|---|---|---|
|use_wandb|Wandb 사용 여부|True|True,False|
|use_cm|Confusion metrix 사용 여부|True|True,False|
|entity|Wandb 엔티티 명|"connect-cv-04"|---|
|project_name|Wandb 프로젝트 명|"Image_classification_mask"|---|
|exp_name|실험명|"exp"|---|
|exp_num|실험 번호|0|---|
|user_name|실험자|"my_name"|"KDH","KJY","HJH","KDK"|

### 실험 관련 설정
|Argument|Description|Default|Possible value|
|---|---|---|---|
|datadir|input 경로|'../input|---|
|train_file|train csv 이름|"train.csv"|---|
|valid_file|valid csv 이름|"valid.csv"|---|
|transform|Transform 목록|['resize','randomrotation', 'totensor', 'normalize']|---|
|seed|Random seed|223|---|
|model_name|Model_names|“CustomModel”|---|
|model_param|Model_names|{pretrained : True, backbone : "resnet18"}|---|
|num_classes|Class 개수|18|---|
|batch_size|Batch size|32|---|
|opt_name|Optimizer|"Adam"|"Adam"|
|loss|loss 종류|"crossentropy"|"crossentropy","focalloss","f1loss","bceloss","mseloss"|
|loss_param|loss parm|"미정"|---|
|lr|learning rate|5e-6|---|
|lr_sheduler|"Learning rate scheduler|"StepLR"|"StepLR","ReduceLROnPlateau"|
|lr_sheduler_param|Lr scheduler parameter|"미정"|---|
|weight_decay|Weight Decay|5e-4|---|
|epochs|epoch|100|---|
|savedir|모델 저장 위치|"./checkpoint"|---|
|grad_accum_steps|---|1|---|
|mixed_precision|---|"fp16"|---|
|patience|Early Stopping|100|---|


# Contributors ✨
