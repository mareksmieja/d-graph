# d-graph

This is a Matlab demo implementation of d-graph, a semi-supervised clustering method published in the paper TODO. It takes pairsise constraints and unlabeled data and returns a partition, which is both consistent with the side information as well as with the internal structure of data. 

Example files are placed in "data" subdirectory while Matlab code can be found in "code" directory. The main file for linear kernel is "runner_CV.m" while the kernel version can be run with "runner_CV_Kernel.m". The results will be put into "data/res" directory. The code requires minFunc library from https://www.cs.ubc.ca/~schmidtm/Software. 

For any questions or assist, email me at smieja.marek [at] gmail.com. 
