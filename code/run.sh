echo "==MetaProx MAIN=="

####################
# Classification
####################

#nohup python MetaProxMain.py --gpu_id 0 --seed 101 --expt 5way1shot --ds MINI_IMAGENET --backbone CONV4 --method MetaProx &
#nohup python MetaProxMain.py --gpu_id 0 --seed 101 --expt 5way5shot --ds MINI_IMAGENET --backbone CONV4 --method MetaProx &

####################
# Regression
####################

##########
# sine
##########
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 2shot --ds SINE --backbone MLP2 --noise_sigma 0 &
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 2shot --ds SINE --backbone MLP2 --noise_sigma 1 &
nohup python MetaProxMain.py --gpu_id 7 --cls_or_reg reg --expt 5shot --ds SINE --backbone MLP2 --noise_sigma 0 &
nohup python MetaProxMain.py --gpu_id 7 --cls_or_reg reg --expt 5shot --ds SINE --backbone MLP2 --noise_sigma 1 &

##########
# Sale
##########
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 1shot --ds SALE --backbone MLP2 &
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 5shot --ds SALE --backbone MLP2 &

##########
# QMUL
##########
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 10shot --ds QMUL --backbone CONV3 &
#nohup python MetaProxMain.py --gpu_id 0 --cls_or_reg reg --expt 10shot --ds QMUL --backbone CONV3 --is_out_range True &


