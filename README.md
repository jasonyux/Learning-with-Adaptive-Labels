# Learning with Adaptive Labels

This is an implementation of the paper (archiv link here when available).

## Project Layout
```bash
root
├── algorithms # algorithm implementations, for LwAL, LWR, etc.
├── datasets # processed datasets used in this project
├── experiments # code for reproducing the experiments
├── outputs # stores graph outputs, such as Dendrograms
└── utils # utility code used by `algorithms` and `experiments`
```

## Datasets

Datasets used in the paper (downloaded as of Aug 2022).

Image Datasets:
- [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
- [CIFAR10](https://www.tensorflow.org/datasets/catalog/cifar10)
- [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)
- [FOOD101](https://www.tensorflow.org/datasets/catalog/food101)
- [AwA2](https://cvml.ist.ac.at/AwA2/)

Text Datasets:
- [IMDB-Movie-Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)
- [Yelp-Polarity-Reviews](https://www.tensorflow.org/datasets/catalog/yelp_polarity_reviews)

### Dataset Preprocessing
We mostly only saved the datasets into relevant folders under the `datasets` folder, so that python files in `experiments` can easily access the datasets. 

Since this usually invovles downloading large files, we provide examples of how to process the datasets in the `data_preprocessing.ipynb` file, so that you can choose only to process/download the datasets you want to test.

## Architectures

Backbones used in the paper (downloaded as of Aug 2022)

For image datasets:
- [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50)
- [EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet/EfficientNetB0)
- [DenseNet121](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121)

For text dataset:
- [BERT](https://huggingface.co/docs/transformers/model_doc/bert)

## Usages

We provide examples and explainations of how to reproduce the results in the paper.

## Learning Speed and Test Performance
To reproduce the learning speed and test performance results of different algorithms, run the following code under the root directory of this project:

- experiments with image datasets:
	```bash
	python experiments/learning_speed_exp.py
	```
- experiments with text datasets:
	```bash
	python experiments/text_exp.py
	```

We placed all the configuration files in the python file. For instance, in the `experiments/learning_speed_exp.py` file, you will see:
```python
CONFIG = {
	"dataset": {
		"name": "cifar10", # ["mnist", "fashion_mnist", "cifar10", "cifar100", "food101"]
		"num_labels": 10, # [10, 10, 10, 100, 101]	
	},
	"model": {
		"encoder": "densenet", # ["resnet50", "efficientnet", "densenet"]
		"algo": 'lwal', # ["std", "staticlabel", "lwr", "label_embed", "lwal"]
		"rloss": "cos_repel_loss_z", # ["cos_repel_loss_z", "none"]
		"latent_dim": 100, # num_labels for normal algorithms, 10 * num_labels for LwAL10, 768 for StaticLabel
		"stationary_steps": 1, # used by LwAL
		"warmup_steps": 0, # [0, 2, 5] for LwAL
		"k": 5, # [2, 3, 5] for LWR
	},
	"training": {
		"lr": 1e-4, # [1e-4, 1e-3]. 1e-3 used only for efficientnet
		"epochs": 10, # [10, 20]. 20 used for large datasets (CIFAR100, Food101)
	},
	"seed": 123 # [12, 123, 1234]
}
```
this means that, if you want to test, say `lwr` algorithm with `k=2`, using `resnet50` encoder on the `cifar100` dataset over `20` epochs, you can change the configuration to
```python
CONFIG = {
	"dataset": {
		"name": "cifar100", # ["mnist", "fashion_mnist", "cifar10", "cifar100", "food101"]
		"num_labels": 100, # [10, 10, 10, 100, 101]	
	},
	"model": {
		"encoder": "resnet50", # ["resnet50", "efficientnet", "densenet"]
		"algo": 'lwr', # ["std", "staticlabel", "lwr", "label_embed", "lwal"]
		"rloss": "none", # ["cos_repel_loss_z", "none"]
		"latent_dim": 100, # num_labels for normal algorithms, 10 * num_labels for LwAL10, 768 for StaticLabel
		"stationary_steps": 1, # used by LwAL
		"warmup_steps": 0, # [0, 2, 5] for LwAL
		"k": 2, # [2, 3, 5] for LWR
	},
	"training": {
		"lr": 1e-4, # [1e-4, 1e-3]. 1e-3 used only for efficientnet
		"epochs": 20, # [10, 20]. 20 used for large datasets (CIFAR100, Food101)
	},
	"seed": 123 # [12, 123, 1234]
}
```
and it should run accordingly.

### Semantic Label Representation

To reproduce the Dendrograms and the correlation scores, you can run the following code under the root project:

```bash

python experiments/hierarchical_exp.py
```
Note that 

- running this will save the graphs to the `outputs` folder under this project. 
- experimental configuration is stored in the `experiments/hierarchical_exp.py` file, and customizatons can be done by following the examples mentioned in the previous section.
