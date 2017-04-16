This code includes the detailed implementation of the paper:

Reference:
Hu, Y., Zhang, D., Ye, J., Li, X., & He, X. (2013). Fast and accurate matrix 
completion via truncated nuclear norm regularization. IEEE Transactions on 
Pattern Analysis and Machine Intelligence, 35(9), 2117-2130.

The code contains:
|--------------
|-- TNNR_main.m           entrance to start the experiment
|-- pic/                  directory for original images

|-- function/                 functions of TNNR-WRE algorithm
    |-- PSNR.m                compute the PSNR and Erec for recovered image
    |-- TNNR_WRE_algorithm.m  main part of TNNR-WRE implementation
    |-- weight_matrix.m       compute weight matrix in an increasing order
    |-- weight_sort.m         sort the sequence of weight value according to
                                  observed elements; rows with more observed 
                                  elements are given smaller weights
|-- image/                    directory for original images
|-- mask/                     directory for various mask types, 300x300
|-- result/                   directory for saving experimental results
|-------------

For algorithm interpretation, please read Liu et al. (2016) paper and Hu et 
al. (2013) paper, in which more details are demonstrated.

If you have any questions about this implementation, please do not hesitate 
to contact me.

Xue Shengke, 
College of Information Science and Electronic Engineering,
Zhejiang University, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, or xueshengke1993@gmail.com.