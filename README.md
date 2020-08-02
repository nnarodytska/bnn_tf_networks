## To load model 

## Original BNN 

python3.6 tf_binary_load.py --load  ./mnist/500_bin_0_1/


python3.6 tf_binary_load.py --load  ./fashion/500_bin_0_1/


## Sparse BNN 

python3.6 tf_binary_load.py --load  ./mnist/500_quant_0_1/


python3.6 tf_binary_load.py --load  ./fashion/500_quant_0_1/

## Sparse + Stable BNN 

python3.6 tf_binary_load.py --load  ./mnist/500_quant_0_1_stable_0_1/


python3.6 tf_binary_load.py --load  ./fashion/500_quant_0_1_stable_0_1/


## Sparse + L1  BNN 


python3.6 tf_binary_load.py --load  ./mnist/500_reg_0_1/


python3.6 tf_binary_load.py --load  ./fashion/500_reg_0_1/


## Sparse + L1 +  Stable BNN 


python3.6 tf_binary_load.py --load  ./mnist/500_reg_0_1_stable_0_1/


python3.6 tf_binary_load.py --load  ./fashion/500_reg_0_1_stable_0_1/

