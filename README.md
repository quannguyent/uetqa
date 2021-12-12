# About
.

# Installation
create a new python venv is recommended
```
git clone <this repo>

cd uetqa

pip install -e .
```

# data

index: https://drive.google.com/file/d/19xvIVd_o4_aeGCSXbe3PJcjTPZFzSA7x/view?usp=sharing

reader model: https://drive.google.com/file/d/1i6zl9EN1p9aZmZoVZIqVyPYvcmd7U9sC/view?usp=sharing

# cmd
interactive bertserini
```
python3 -m bertserini.interactive \
  --reader-model /path/to/models/output-xlm-roberta-large \
  --language vi --index-path /path/to/indexes/uet_regulations \
  --n_phrases 5
```

