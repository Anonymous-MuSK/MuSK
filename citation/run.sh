python -u train.py --data cora --layer 64 --test
python -u s_train.py --data cora --layer 64 --test --lbd_kl 1 --lbd_pr 0.01 --kernel kl

python -u train.py --data citeseer --layer 64 --hidden 256 --lamda 0.6 --dropout 0.7 --test
python -u s_train.py --data citeseer --layer 64 --t_hidden 256 --s_hidden 256 --lamda 0.6 --dropout 0.7 --test --lbd_kl 0.1 --lbd_pr 0.01 --kernel kl

python -u train.py --data pubmed --layer 64 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test
python -u s_train.py --data pubmed --layer 64 --hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test --lbd_kl 100 --lbd_pr 10 --kernel kl
