# Local-Semantic-Feature-Aggregation-Based-Transformer-for-Hyperspectral-Image-Classification
Local Semantic Feature Aggregation-Based Transformer for Hyperspectral Image Classification

The code in this toolbox implements the ["Local Semantic Feature Aggregation-Based Transformer for Hyperspectral Image Classification"](https://ieeexplore.ieee.org/document/9864609). The code may easy to understand. 

How to use it?
---------------------
Here an example experiment is given by using **Pavia University hyperspectral data**. Directly run **main.py** functions with different network parameter settings to produce the results. Please note that due to the randomness of the parameter initialization, the experimental results might have slightly different from those reported in the paper. You can adjust the network in **model.py** functions.

Dataset path
---------------------
 ```
|--main.py
|--model.py
|--data
|  |--PaviaU.mat
|  |--PaviaU_gt.mat
```

System-specific notes
---------------------
The codes of networks were tested using PyTorch 1.9 version (CUDA 10.2) in Python 3.7 on Window 10 system.

Contact Information:
--------------------
If you encounter the bugs while using this code, please do not hesitate to contact us.
Bing Tu : tubing@nuist.edu.cn

Citation
---------------------

**Please kindly cite the papers if this code is useful and helpful for your research.**

Bing Tu, Xiaolong Liao, Qianming Li, Yishu Peng and Antonio Plaza, "Local Semantic Feature Aggregation-Based Transformer for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5536115, doi: 10.1109/TGRS.2022.3201145.
```
  @article{Tu2022LSFAT,
    author={Tu, Bing and Liao, Xiaolong and Li, Qianming and Peng, Yishu and Plaza, Antonio},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={Local Semantic Feature Aggregation-Based Transformer for Hyperspectral Image Classification}, 
    year={2022},
    volume={60},
    number={},
    pages={1-15},
    doi={10.1109/TGRS.2022.3201145}}
```
Acknowledgement
---------------------

In addition, we are also very grateful to Professor Sun Le's code for providing us with principal component analysis implement and some ideas. His wonderful work as follows:
```
@article{Sun2022SSFTT,
  author={Sun, Le and Zhao, Guangrui and Zheng, Yuhui and Wu, Zebin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Spectral–Spatial Feature Tokenization Transformer for Hyperspectral Image Classification}, 
  year={2022},
  volume={60},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2022.3144158}}
```
