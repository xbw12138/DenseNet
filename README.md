# DenseNet

## Install
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install torchsummary
pip install matplotlib
pip install tqdm
pip install sklearn
```

## Datasets

* Photos
    * Train
        * Class-1
        * Class-2
    * Test
        * Class-1
        * Class-2
    * Validation
        * Class-1
        * Class-2


## Usage
* Train

```
python main.py --num_classes 2 --dataset_validation Photos/Validation --dataset_test Photos/Test --dataset_train Photos/Train --size 224 train
```

* Test

```
python main.py --num_classes 2 --dataset_validation Photos/Validation --dataset_test Photos/Test --dataset_train Photos/Train --size 224
--ckpt model.pth test
```