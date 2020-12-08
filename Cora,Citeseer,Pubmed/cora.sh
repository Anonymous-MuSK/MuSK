#!/bin/bash

for i in {1..10}
	do python -u s_train_plus.py --data cora --layer 64 --test --dev 1 --lbd_kl 1 --lbd_pr 1 --kernel kl --seed $i
	#do python -u train.py --data cora --layer 64 --test --dev 3 --seed $i
	#do python -u s_kd.py --data cora --layer 64 --test --dev 1 --lbd_kl 0.01 --seed $i
	#do python -u s_lsp.py --data cora --layer 64 --test --dev 1 --lbd_lsp 10 --seed $i
done


