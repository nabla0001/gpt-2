# Building GPT-1 in PyTorch

The goal of this project is to implement the first [**Generative Pre-trained Transformer (GPT)**](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
(Radford et al. 2018) aka **GPT-1** from the ground up in PyTorch and train it on a real-world dataset.


We will train it on [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/), 
an open-source approximation of OpenAI's proprietary WebText dataset which was used to train GPT-2.

This dataset is sufficiently large to expose us to some of the challenges of large-scale training and challenge us to care about
efficiency in tokenization, data-processing and model. It will (hopefully) also result in a more interesting language model 
than if we only used toy data. 
At the same time, it is possible to train on this data on consumer-grade hardware.


The code is based on Andrej Karpathy's fantastic [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) but aims to 
be more readable and focused on re-producing (Radford et al. 2018). This means that a good chunk of additional functionality
Andrej is building in his repo is dropped to hopefully make the implementation easy to follow.

## References

[1] A. Radford, K.Narasimhan, T. Salimans, I. Sutskever. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). 2018.

[2] A. Gokaslan, V. Cohen. [OpenWebText Corpus](http://Skylion007.github.io/OpenWebTextCorpus). 2019.

