# Sample photos for the demo

Shown in the UI under **Try a sample** (and via `?sample=<name>` links, e.g. `http://localhost:8501/?sample=leaf_rust`). Each was chosen by running the release bundle so it demonstrates one behaviour.

| File | Shows | Source / licence |
|---|---|---|
| `healthy.jpg`, `cercospora.jpg`, `leaf_rust.jpg`, `phoma.jpg` | a confident, correct diagnosis | [Ethiopian Coffee Leaf Disease dataset](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease) (CC0), test split |
| `uncertain.jpg` | an *uncertain* answer (Cercospora 74% / Leaf Rust 25%) | same dataset (CC0), test split |
| `too_dark.jpg` | a retake request (too dark) | `healthy.jpg` darkened to 18% brightness |
| `not_coffee_bean_leaf.jpg` | an out-of-distribution rejection | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) (MIT licence) |

Images are the 384 px versions used by the pipeline. `samples.json` holds the titles and true classes.
