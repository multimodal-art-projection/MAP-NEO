# MAP-NEO: A fully open-sourced Large Language Model


## Introduction
MAP-NEO is a fully open-sourced Large Language Model that includes the pretraining data, a data processing pipeline (Matrix), pretraining scripts, and alignment code. It is trained from scratch on 4.5T English and Chinese tokens, exhibiting performance comparable to LLaMA2 7B. The MAP-Neo model delivers proprietary-model-like performance in challenging tasks such as reasoning, mathematics, and coding, outperforming its peers of similar size. For research purposes, we aim to achieve full transparency in the LLM training process. To this end, we have made a comprehensive release of MAP-Neo, including the final and intermediate checkpoints, a self-trained tokenizer, the pre-training corpus, and an efficient, stable optimized pre-training codebase.

## Model & DATA Downloads

We release the MAP-NEO 7B, including both base and chat models, to the public. To support a broader and more diverse range of research within both academic and commercial communities. Please **note** that the use of this model is subject to the terms outlined in [License section](#license). Commercial usage is permitted under these terms.

### Huggingface

|         Model         |                                 Download                                 |
|:---------------------:|:-----------------------------------------------------------------------:|
| MAP-NEO 7B Base       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_7b)  |
| MAP-NEO 7B intermedia       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_7b_intermediate)  |
| MAP-NEO 7B decay       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_7b_decay)  |
| MAP-NEO 2B Base       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_2b_general)  |
| MAP-NEO scalinglaw 980M       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_scalinglaw_980M)  |
| MAP-NEO scalinglaw 460M       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_scalinglaw_460M)  |
| MAP-NEO scalinglaw 250M       | 🤗 [HuggingFace](https://huggingface.co/m-a-p/neo_scalinglaw_250M)  |
| MAP-NEO DATA Matrix   | 🤗 [HuggingFace](https://huggingface.co/datasets/m-a-p/Matrix)  |
## License
This code repository is licensed under the MIT License. The use of MAP-Neo models is subject to the Model License. MAP-Neo supports commercial use.
