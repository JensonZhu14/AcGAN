## Look globally, age locally: Face aging with an attention mechanism (AcGAN)

PyTorch implementation of the AcGAN algorithm. 

### 1. The Architecture of AcGAN



<img src="https://github.com/JensonZhu14/AcGAN/tree/master/images/face_aging_network.png" width="500" alt="note"/>

![Architecture of AcGAN](https://github.com/JensonZhu14/AcGAN/tree/master/images/face_aging_network.png)

### 2.Prerequisites

* Python 3.6

* PyTorch 1.3.0
* GPU

### Dataset & Preparation

* [Morph](https://ebill.uncw.edu/C20231_ustores/web/classic/product_detail.jsp?PRODUCTID=8)
* [CACD](http://bcsiriuschen.github.io/CARC/_)

### Training

Training a model by:

```
$ python main.py config/morph.yml
```

### Results

* Attention Results

![attention_results](https://github.com/JensonZhu14/AcGAN/tree/master/images/attention_result.png)

* Results on Morph Dataset

  ![results_on_morph](https://github.com/JensonZhu14/AcGAN/tree/master/images/aging_morph_result.png)

* Comparison of AcGAN, IPCGAN, and CAAE in Morph Dataset

  ![comparison_result](https://github.com/JensonZhu14/AcGAN/tree/master/images/comparison_in_vis.png)

  

### Citation



### License

**AcGAN** is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).