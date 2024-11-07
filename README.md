# Building GPT-1 in PyTorch

The goal of this project is to implement the first [**Generative Pre-trained Transformer (GPT)**](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
(Radford et al. 2018) aka **GPT-1** from the ground up in PyTorch and train it on a real-world dataset.


We will train it on [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/), 
an open-source approximation of OpenAI's proprietary WebText dataset which was used to train GPT-2 (Radford et al 2019).

This dataset is sufficiently large to expose us to some of the challenges of large-scale training and challenge us to care about
efficiency in tokenization, data-processing and model. It will (hopefully) also result in a more interesting language model 
than if we only used toy data. 
At the same time, it is possible to train on this data on consumer-grade hardware.


## Acknowledgement

The code is strongly inspired by Andrej Karpathy's fantastic [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) but aims to 
focus entirely on understanding and re-producing GPT-1.
This means that a good chunk of additional functionality Andrej is building in his repo to make it applicable to a variety 
of tasks is dropped to hopefully provide a more readable, clean (yet efficient)
implementation for those interested in the original paper.


## References

[1] A. Radford, K.Narasimhan, T. Salimans, I. Sutskever. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). 2018.

[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). 2019. 

[3] A. Gokaslan, V. Cohen. [OpenWebText Corpus](http://Skylion007.github.io/OpenWebTextCorpus). 2019.

[4] D. Jurafsky, J. Martin. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf). 2024.
