#!/bin/bash

for i in {1..1}
    #do python -u train.py --data pubmed --layer 1 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --dev 1 --seed $i
    do python -u s_train.py --data pubmed --layer 64 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --dev 0 --lbd_kl 100 --lbd_pr 10 --seed $i
    #do python -u s_kd.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --dev 0 --lbd_kl 0.1 --seed $i
    #do python -u s_lsp.py --data pubmed --layer 16 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --dev 0 --lbd_lsp 10 --seed $i
done


