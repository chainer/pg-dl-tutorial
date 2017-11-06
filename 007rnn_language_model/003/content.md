1.2 Perplexity (Evaluation of the language model)
-----------------------------------------------

**Perplexity** is the common evaluation metric for a language model. Generally, it measures how well the proposed probability model $P_{\rm model}({\bf X})$ represents the target data $P^*({\bf X})$.

Let a validation dataset be $D = \{{\bf X}^{(n)}\}_{n=1}^{|D|}$, which is a set of sentences, where the $n$-th sentence length is $T^{(n)}$, and the vocabulary size of this dataset is $|\mathcal{V}|$, the perplexity is represented as follows:

$$
\begin{eqnarray}
&& b^z \\
&& s.t.~ ~ ~ z = - \frac{1}{|\mathcal{V}|}
\sum_{n=1}^{|D|} \sum_{t=1}^{T^{(n)}} \log_b P_{\rm model}({\bf x}_t^{(n)}, {\bf X}_{[a, t-1]}^{(n)})
\end{eqnarray}
$$

We usually use $b = 2$ or $b = e$. The perplexity shows how much varied the predicted distribution for the next word is. When a language model well represents the dataset, it should show a high probability only for the correct next word, so that the entropy should be high. In the above equation, the sign is reversed, so that smaller perplexity means better model.

During training, we minimize the below cross entropy:

$$
\mathcal{H}(\hat{P}, P_{\rm model}) = - \hat{P}({\bf X}) \log P_{\rm model}({\bf X})
$$

where $\hat P$ is the empirical distribution of a sequence in the training dataset.
