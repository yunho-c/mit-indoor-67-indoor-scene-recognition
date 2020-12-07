# [MIT-Indoor-67 Dataset](http://web.mit.edu/torralba/www/indoor.html)

The [MIT Indoor Scene 67 dataset](http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar), developed in MIT, contains 67 labelled categories with 15620 images. Following a 80:20 split, a subset was splited into the [train](http://web.mit.edu/torralba/www/TrainImages.txt) and [test](http://web.mit.edu/torralba/www/TestImages.txt) sets with labels and each class contains the same number of train and test sets. 

![MITIndoor67](http://web.mit.edu/torralba/www/allIndoors.jpg)

### Indoor Scene Classification
This project is original the capstone of Udacity's Machine Learning Engineer Nanodegree. The goal was to train an EfficientNet model to classify indoor scene images with the MIT Indoor 67 dataset.


The classification model was developed on AWS SageMaker and Pytorch framework. You can find the steps and process of developing an EfficientNet model of scene classification with the following notebooks:

#### [1.  Data Exploration](./notebooks/ENindoor67-Exploration.ipynb)
#### [2. Data Preprocessing](./notebooks/ENindoor67-Preprocessing.ipynb)
#### [3. Benchmark Model](./notebooks/ResNeXt101.ipynb)
#### [4. EfficientNet Base Model](./notebooks/EfficientNets-Base.ipynb)
#### [5. Fine-tuning and Hyperparameter Tuning by Bayesian Search](./notebooks/EfficientNets-HPO.ipynb)
#### [6. Model testing](./notebooks/ENindoor67-LocalTesting.ipynb)

### Retrieving MIT Indoor 67 Dataset

To retrieve the dataset (.tar) and the subset split labels (.txt), run the following cell in a jupyter notebook (.ipynb):


```python
!mkdir -p data/mit_indoor_67/raw
!wget http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
!wget http://web.mit.edu/torralba/www/TrainImages.txt -P data/mit_indoor_67
!wget http://web.mit.edu/torralba/www/TestImages.txt -P data/mit_indoor_67
!tar -xf indoorCVPR_09.tar -C data/mit_indoor_67/raw
!rm -rf indoorCVPR_09.tar
```

    --2020-11-28 23:31:55--  http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar
    Resolving groups.csail.mit.edu (groups.csail.mit.edu)... 128.30.2.44
    Connecting to groups.csail.mit.edu (groups.csail.mit.edu)|128.30.2.44|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 2592010240 (2.4G) [application/x-tar]
    Saving to: ‘indoorCVPR_09.tar’
    
    indoorCVPR_09.tar   100%[===================>]   2.41G  9.54MB/s    in 3m 27s  
    
    2020-11-28 23:35:23 (11.9 MB/s) - ‘indoorCVPR_09.tar’ saved [2592010240/2592010240]
    
    --2020-11-28 23:35:23--  http://web.mit.edu/torralba/www/TrainImages.txt
    Resolving web.mit.edu (web.mit.edu)... 104.100.30.13, 2600:1408:8400:58e::255e, 2600:1408:8400:5ab::255e
    Connecting to web.mit.edu (web.mit.edu)|104.100.30.13|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 172080 (168K) [text/plain]
    Saving to: ‘data/mit_indoor_67/TrainImages.txt.1’
    
    TrainImages.txt.1   100%[===================>] 168.05K  --.-KB/s    in 0.07s   
    
    2020-11-28 23:35:23 (2.38 MB/s) - ‘data/mit_indoor_67/TrainImages.txt.1’ saved [172080/172080]
    
    --2020-11-28 23:35:23--  http://web.mit.edu/torralba/www/TestImages.txt
    Resolving web.mit.edu (web.mit.edu)... 104.100.30.13, 2600:1408:8400:5ab::255e, 2600:1408:8400:58e::255e
    Connecting to web.mit.edu (web.mit.edu)|104.100.30.13|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 43475 (42K) [text/plain]
    Saving to: ‘data/mit_indoor_67/TestImages.txt.1’
    
    TestImages.txt.1    100%[===================>]  42.46K  --.-KB/s    in 0.01s   
    
    2020-11-28 23:35:23 (2.83 MB/s) - ‘data/mit_indoor_67/TestImages.txt.1’ saved [43475/43475]
    


Let's have an overview of the dataset:


```python
import os
from glob import glob

data_dir = 'data/mit_indoor_67/raw/Images'
category_dir = glob(os.path.join(data_dir, '*'))
image_files = []
for category in category_dir:
    image_files += glob(os.path.join(category, '*'))
    
print(f"""Number of categories: {len(category_dir)}
Number of Images: {len(image_files)}""")
```

    Number of categories: 67
    Number of Images: 15620


Let's have a look at the train-test split in the [original paper](http://people.csail.mit.edu/torralba/publications/indoor.pdf)


```python
from collections import Counter
from heapq import nlargest, nsmallest

train_txt = 'data/mit_indoor_67/TrainImages.txt'
test_txt = 'data/mit_indoor_67/TestImages.txt'

train_image_paths = open(train_txt).read().split('\n')
test_image_paths = open(test_txt).read().split('\n')

train_image_counts = Counter([path.split('/')[0] for path in train_image_paths])
test_image_counts = Counter([path.split('/')[0] for path in test_image_paths])

print(f"""TRAINING SET:
Number of samples: {len(train_image_paths)}
Number of categories: {len(train_image_counts)}
Subsample range of each category (min. - max.): {train_image_counts[nsmallest(1, train_image_counts, key=train_image_counts.get)[0]]} - {train_image_counts[nlargest(1, train_image_counts, key=train_image_counts.get)[0]]}""")

print(f"""TEST SET:
Number of samples: {len(test_image_paths)}
Number of categories: {len(test_image_counts)}
Subsample range of each category (min. - max.): {test_image_counts[nsmallest(1, test_image_counts, key=test_image_counts.get)[0]]} - {test_image_counts[nlargest(1, test_image_counts, key=test_image_counts.get)[0]]}""")
```

    TRAINING SET:
    Number of samples: 5360
    Number of categories: 67
    Subsample range of each category (min. - max.): 77 - 83
    TEST SET:
    Number of samples: 1340
    Number of categories: 67
    Subsample range of each category (min. - max.): 17 - 23


The original study used only a subset of the sample. In order to train our model with a larger dataset, we will follow the study's 80:20 split on the whole dataset.

Finally, let's have a look at the size of the first 10 images in both sets:


```python
import cv2

def print_shape(data_dir, paths):
    print()
    for idx, path in enumerate(paths):
        img = cv2.imread(os.path.join(data_dir, path))
        print(f'{idx+1} - {path} is of size: {img.shape[0]} x {img.shape[1]}')

print("FIRST 10 TRAIN IMAGE SIZE:")
print_shape(data_dir, train_image_paths[:10])
print()
print("FIRST 10 TEST IMAGE SIZE:")
print_shape(data_dir, test_image_paths[:10])
```

    FIRST 10 TRAIN IMAGE SIZE:
    
    1 - gameroom/bt_132294gameroom2.jpg is of size: 296 x 397
    2 - poolinside/inside_pool_and_hot_tub.jpg is of size: 412 x 550
    3 - winecellar/bodega_12_11_flickr.jpg is of size: 375 x 500
    4 - casino/casino_0512.jpg is of size: 258 x 400
    5 - livingroom/living58.jpg is of size: 768 x 1024
    6 - mall/4984307.jpg is of size: 1261 x 1280
    7 - corridor/pasilltmpo_t.jpg is of size: 256 x 257
    8 - laboratorywet/laboratorio_quimica_07_05_altavista.jpg is of size: 296 x 396
    9 - bookstore/CIMG2743.jpg is of size: 336 x 448
    10 - casino/casino_0044.jpg is of size: 266 x 400
    
    FIRST 10 TEST IMAGE SIZE:
    
    1 - kitchen/int474.jpg is of size: 256 x 256
    2 - operating_room/operating_room_31_03_altavista.jpg is of size: 209 x 260
    3 - restaurant_kitchen/restaurant_kitchen_google_0075.jpg is of size: 351 x 470
    4 - videostore/videoclub_05_14_flickr.jpg is of size: 500 x 375
    5 - poolinside/piscine_interieureee.jpg is of size: 369 x 460
    6 - videostore/blockbuster_08_10_flickr.jpg is of size: 415 x 500
    7 - poolinside/piscina_cubierta_07_19_altavista.jpg is of size: 225 x 300
    8 - mall/mall26.jpg is of size: 256 x 256
    9 - kindergarden/toddler.jpg is of size: 376 x 628
    10 - buffet/Buffet_1.jpg is of size: 503 x 668


The images seem to have various size. Hence, we may want to resize the images in our pre-processing stage.

Let's first of all [explore our data](notebooks/ENindoor67-Exploration.ipynb) for exploration, visualization and preprocessing.


```python

```
