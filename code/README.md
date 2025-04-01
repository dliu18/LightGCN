```
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64
python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" --topks="[20]" --recdim=64
```

```
python read_models.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64
python read_models.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="yelp2018" --topks="[20]" --recdim=64
python read_models.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="amazon-book" --topks="[20]" --recdim=64
```

python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20, 1000]" --recdim=64 --tau=10 --use_cpp=1 --shuffle_users=0 --item_pairs=1 --comment="tau/10"

## Assumptions

Assumes that the training and testing data have interactions for each user.
The user ids are exactly {0, ..., n_user - 1} with no ommissions