## Look globally, age locally: Face aging with an attention mechanism (AcGANs)

PyTorch implementation of the AcGANs algorithm. 

### 1. The Architecture of AcGANs

---



![Architecture of AcGAN](images/face_aging_network.png)

### 2.Prerequisites

----



* Python 3.6

* PyTorch 1.3.0
* GPU

### Dataset & Preparation

------



* [Morph](https://ebill.uncw.edu/C20231_ustores/web/classic/product_detail.jsp?PRODUCTID=8)
* [CACD](http://bcsiriuschen.github.io/CARC/_)

### Training

----



Training a model by:

```
$ python main.py config/morph.yml
```

### Results

-----



* Attention Results
* ![attention_results](images/attention_result.png)

* Results on the Morph Dataset

* ![results_on_morph](images/aging_morph_result.png)

* Comparison of AcGANs, IPCGANs, and CAAE in the Morph Dataset

  ![comparison_result](images/comparison_in_vis.png)

  

### Citation

-----





### License

------



**AcGANs** is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, contact [Junping Zhang](http://www.pami.fudan.edu.cn/~jpzhang/).