# Building GPT-2 in PyTorch

The goal of this project is to re-create the second [**Generative Pre-trained Transformer (GPT)**](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) aka 
**GPT-2** (Radford et al. 2019) from the ground up in PyTorch.


We will train it on [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/), 
an open-source approximation of OpenAI's proprietary WebText dataset which was used in the original paper.
This dataset is sufficiently large to expose us to some of the challenges of large-scale training and challenge us to care about
efficiency in tokenization, data-processing and model.
At the same time, it is possible to train on this data on consumer-grade hardware (🤞).


## Acknowledgement

The code is strongly inspired by Andrej Karpathy's fantastic [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) but aims to 
focus entirely on understanding and re-producing GPT-2.
This means that a good chunk of additional functionality Andrej is building in his repo to make it applicable to a variety 
of tasks like fine-tuning is dropped to hopefully provide a more readable, clean (yet efficient)
implementation for those interested in the original paper.


## References

[1] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 2019. 

[2] A. Gokaslan, V. Cohen. [OpenWebText Corpus](http://Skylion007.github.io/OpenWebTextCorpus). 2019.
