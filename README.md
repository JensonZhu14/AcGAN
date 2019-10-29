## Look globally, age locally: Face aging with an attention mechanism (AcGANs)

PyTorch implementation of the AcGANs algorithm in the paper ``[Look globally, age locally: Face aging with an attention mechanism.](http://arxiv.org/abs/1910.12771)''. 

### 1. The Architecture of AcGANs

---



![Architecture of AcGAN](images/face_aging_network.png)

### 2. Prerequisites

----



* Python 3.6

* PyTorch 1.3.0
* GPU

### 3. Dataset & Preparation

------



* [Morph](https://ebill.uncw.edu/C20231_ustores/web/classic/product_detail.jsp?PRODUCTID=8)
* [CACD](http://bcsiriuschen.github.io/CARC/_)

### 4. Training

----



Training a model by:

```
$ python main.py config/morph.yml
```

### 5. Results

-----



* Attention Results
* ![attention_results](images/attention_result.png)

* Results on the Morph Dataset

* ![results_on_morph](images/aging_morph_result.png)

* Comparison of AcGANs, IPCGANs, and CAAE in the Morph Dataset

  ![comparison_result](images/comparison_in_vis.png)

  

### 6. Citation

-----

*Haiping Zhu, Zhizhong Huang, Hongming Shan, Junping Zhang*. Look globally, age locally: Face aging with an attention mechanism. arXiv:1910.12771, 2019.



### 7. License

------



**AcGANs** is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).