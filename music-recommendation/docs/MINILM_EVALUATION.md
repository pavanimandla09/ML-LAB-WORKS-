# MiniLM semantic recommender evaluation

SoundScout now has an optional offline semantic recommender using the pretrained
`sentence-transformers/all-MiniLM-L6-v2` encoder. It embeds the available
track title, artist, album, genre, mood and tags, then ranks tracks by cosine
similarity to a weighted profile of the user's positive training interactions.

This is an experimental evaluation path. It does not change the deployed API,
website, playback, login, or default evaluation workflow, and it does not
guarantee a higher score.

## Install (one time)

Run from the `music-recommendation` directory:

```powershell
py -3.12 -m pip install -r requirements-ml.txt
```

The first evaluation downloads the pretrained model from Hugging Face and
encodes the supplied catalogue. The model is then cached by the sentence
transformers library.

## Evaluate on the same Last.fm quick sample

```powershell
py -3.12 scripts/evaluate.py --tracks reports/quick_tracks.csv --interactions reports/quick_interactions.csv --dataset-name "Last.fm quick sample" --k 10 --include-minilm --output reports/minilm-evaluation.json
```

The report compares popularity, content, SVD, KNN, hybrid and MiniLM using
the same chronological holdout and sampled candidate construction. Compare
MiniLM primarily on NDCG@10, Recall@10, MAP@10 and Precision@10. The reported
balanced sampled-pair accuracy is a ranking proxy, not conventional
classification accuracy or measured user satisfaction.

Do not change the metric definition or test split to chase a target score.
If MiniLM does not beat the current content baseline, keep the current model
as the stronger result. The live application remains unchanged until the
experiment shows a repeatable improvement on a larger held-out dataset.
