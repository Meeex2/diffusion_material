Run the follwing command for generation:

```
python sample.py --dataname adult --device cuda:0 --steps 50 --save_path sample.csv --inverse_path reconstructed_latents.npy
```

For eval, we can run:

```
python tabsyn/evaluate.py --dataname adult --real_path data/adult/train.csv --synthetic_path sample.csv
```
 For the recovery test:

```

python tabsyn/noise_recovery_test.py --dataname adult --device cuda:0 --num_samples 100 --batch_size 32 --num_steps 20 --save_dir noise_recovery_test
```
For the test:

```
python tabsyn/noise_recovery_test.py \
    --dataname adult \
    --device cuda:0 \
    --num_samples 100 \
    --batch_size 32 \
    --num_steps 20 \
    --save_dir noise_recovery_test
```
