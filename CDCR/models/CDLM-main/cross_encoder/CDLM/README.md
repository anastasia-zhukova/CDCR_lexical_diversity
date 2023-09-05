---
language: en
tags:
- longformer
- cdlm
license: apache-2.0
inference: false

---


# Cross-Document Language Modeling

CDLM: Cross-Document Language Modeling. 
Avi Caciularu, Arman Cohan, Iz Beltagy, Matthew E Peters, Arie Cattan and Ido Dagan. In EMNLP Findings, 2021. [PDF](https://arxiv.org/pdf/2101.00406.pdf)


Please note that during our pretraining we used the document and sentence separators, which you might want to add to your data. The document and sentence separators are `<doc-s>`, `</doc-s>` (the last two tokens in the vocabulary), and `<s>`, `</s>`, respectively.


```python
from transformers import AutoTokenizer, AutoModel
# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('biu-nlp/cdlm')
model = AutoModel.from_pretrained('biu-nlp/cdlm')
```

The original repo is [here](https://github.com/aviclu/CDLM).

If you find our work useful, please cite the paper as:

```python
@article{caciularu2021cross,
  title={Cross-Document Language Modeling},
  author={Caciularu, Avi and Cohan, Arman and Beltagy, Iz and Peters, Matthew E and Cattan, Arie and Dagan, Ido},
  journal={Findings of the Association for Computational Linguistics: EMNLP 2021},
  year={2021}
}
```