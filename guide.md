Run the follwing command for generation:

```
python sample.py --dataname adult --device cuda:0 --steps 50 --save_path sample.csv --inverse_path reconstructed_latents.npy
```

For eval, we can run:

```
python tabsyn/evaluate.py --dataname adult --real_path data/adult/train.csv --synthetic_path sample.csv
```
