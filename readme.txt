This code includes the detailed implementation of the paper:

Reference:
Hu, Y., Zhang, D., Ye, J., Li, X., & He, X. (2013). Fast and accurate matrix 
completion via truncated nuclear norm regularization. IEEE Transactions on 
Pattern Analysis and Machine Intelligence, 35(9), 2117-2130.

First written by debingzhang, Zhejiang Universiy, November 2012.

The code contains:
|--------------
|-- TNNR_main.m            entrance to start real image experiment
|-- TNNR_synthetic.m       entrance to start synthetic experiment
|-- function
    |-- PSNR.m             compute the PSNR and Erec for recovered image
    |-- nuclear_norm.m     compute the nuclear norm of matrix
|-- TNNR-admm/             optimization via ADMM
    |-- result/            directory for saving experimental results
    |-- admm_pic.m         optimization via ADMM
    |-- admmAXB.m          iteration to sovle ||X||_*
|-- TNNR-apgl/             optimization via APGL
    |-- result/            directory for saving experimental results
    |-- admm_apgl.m        optimization via APGL
    |-- apglAXB.m          iteration to sovle ||X||_*
                                observed elements; rows with more observed 
                                elements are given smaller weights
|-- mask/                  directory for various mask types, 300x300
|-- pic/                   directory for original images
|-------------

For algorithm interpretation, please read Hu et al. (2013) paper, in which 
more details are demonstrated.

If you have any questions about this implementation, please do not hesitate 
to contact me.

Ph.D. Candidate, Shengke Xue, 
College of Information Science and Electronic Engineering,
Zhejiang University, Hangzhou, P. R. China,
e-mail: (either one is o.k.)
xueshengke@zju.edu.cn, xueshengke1993@gmail.com.