# Write an RNN Language Model

0. Introduction
================

The **language model** is modeling the probability of generating natural language sentences or documents. You can use the language model to estimate how natural a sentence or a document is. Also, with the language model, you can generate new sentences or documents.

Let's start with modeling the probability of generating sentences. We represent a sentence as ${\bf X} = ({\bf x}_0, {\bf x}_1, \dots, {\bf x}_T)$, in which ${\bf x}_t$ is a one-hot vector. Generally, ${\bf x}_0$ is the one-hot vector of **BOS** (beginning of sentence), and ${\bf x}_T$ is that of **EOS** (end of sentence).

A language model models the probability of a word occurance under the condition of its previous words in a sentence. Let ${\bf X}_{[i, j]}$ be $({\bf x}_i, {\bf x}_{i+1}, \dots, {\bf x}_j)$ , the occurrence probability of sentence ${\bf X}$ can be represented as follows:

$$P({\bf X}) = P({\bf x}_0) \prod_{t=1}^T P({\bf x}_t \mid {\bf X}_{[0, t-1]})$$

So, the language model $P({\bf X})$ can be decomposed into word probabilities conditioned with its previous words.

In this notebook, we model $P({\bf x}_t \mid {\bf X}_{[0, t-1]})$ with a recurrent neural network to obtain a language model $P({\bf X})$.
