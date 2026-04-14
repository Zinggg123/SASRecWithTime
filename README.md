update on 05/23/2025: thx to [Wentworth1028](https://github.com/Wentworth1028) and [Tiny-Snow](https://github.com/Tiny-Snow), we have LayerNorm update, for higher NDCG&HR, and here's the [doc](https://github.com/Tiny-Snow/SASRec.pytorch/blob/main/Result_Norm.md)👍.

---

modified based on [paper author's tensorflow implementation](https://github.com/kang205/SASRec), switching to PyTorch(v1.6) for simplicity, fixed issues like positional embedding usage etc. (making it harder to overfit, except for that, in recsys, personalization=overfitting sometimes)

code in `python` folder.

to train:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

just inference:

```
python main.py --device=cuda --dataset=ml-1m --train_dir=default --state_dict_path=[YOUR_CKPT_PATH] --inference_only=true --maxlen=200

```

output for each run would be slightly random, as negative samples are randomly sampled, here's my output for two consecutive runs:

```
1st run - test (NDCG@10: 0.5897, HR@10: 0.8190)
2nd run - test (NDCG@10: 0.5918, HR@10: 0.8225)
```

pls check paper author's [repo](https://github.com/kang205/SASRec) for detailed intro and more complete README, and here's the paper bib FYI :)

## Time-aware parallel long-short architecture

This workspace version extends the original SASRec with a parallel short-term CNN branch and a time-interval-driven dynamic gate.

- Long-term branch: the original causal self-attention stack, used to capture stable user habits.
- Short-term branch: a causal CNN stack, used to capture recent impulses and local pattern shifts.
- Time-aware gate: combines discrete time buckets, continuous interval features, and recent interaction compactness to decide how much each branch contributes at every position.

New training options:

```
--short_num_blocks=2 --short_kernel_size=3 --recent_window=5 --gate_hidden_units=64
```

Example:

```
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda --short_num_blocks=2 --short_kernel_size=3 --recent_window=5
```

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

I see a dozen of citations of the repo🫰, pls use the example bib as below if needed.
```
@online{huang2020sasrec_pytorch,
  author  = {Zan Huang},
  title   = {SASRec.pytorch},
  year    = {2020},
  url     = {https://github.com/pmixer/SASRec.pytorch}
}
```
