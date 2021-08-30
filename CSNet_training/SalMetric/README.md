# SalMetric

### This is the evaluation code for [DSS](https://github.com/Andrew-Qibin/DSS)

### Requirements   
OpenCV (version 3)   
Cyphon

### Install

For python user
```bash
python setup.py build_ext --inplace
```
or
```bash
python setup.py install
```

For C++ user
```cplusplus
mkdir build && cd build
cmake .. && make
```

### Usage

For app user
```bash
./build/salmetric dataset_list [num_thread]
```
dataset_list should have the following format:   
    1_sal.png 1.png   
    2_sal.png 2.png   
    3_sal.png 3.png   
         ...


For python user
```python
import salmetric
salmetric.do_evalution(num_thread, sal_lst, gt_lst)
```
sal_lst: a list that stores all the saliency maps   
gt_lst : a list that stores all the annotation maps
