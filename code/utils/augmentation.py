import albumentations as A  


def create_transforms(augmentation): 
    if augmentation == 'BaseAugmentation': 
        transforms = A.Compose([
            A.Resize(512, 512), 
        ])
    elif augmentation == 'CustomAugmentation': 
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.RandomCrop(1900, 1900, p=0.5),
            A.Resize(1024, 1024), 
            A.ElasticTransform(alpha=1, alpha_affine=100, border_mode=0, p=0.5),
        ])
    else: 
        raise NotImplementedError 
    
    return transforms 