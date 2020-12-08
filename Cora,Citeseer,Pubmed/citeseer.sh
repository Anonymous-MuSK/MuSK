#!/bin/bash

#python -u train.py --data citeseer --layer 2 --hidden 256 --lamda 0.6 --dropout 0.7 --test
for i in {1..50}
	do python -u s_train.py --data citeseer --layer 64 --hidden 256 --lamda 0.6 --dropout 0.7 --test --lbd_kl 0.1 --lbd_pr 0.01 --kernel kl --dev 1 --seed $i
	#do python -u train.py --data citeseer --layer 64 --hidden 256 --lamda 0.6 --dropout 0.7 --test --dev 0 --seed $i
	#do python -u s_kd.py --data citeseer --layer 1 --hidden 256 --lamda 0.6 --dropout 0.7 --test --dev 3 --lbd_kl 0.1 --seed $i
	#do python -u s_lsp.py --data citeseer --layer 2 --hidden 256 --lamda 0.6 --dropout 0.7 --lbd_lsp 10 --test --dev 3 --seed $i
done


