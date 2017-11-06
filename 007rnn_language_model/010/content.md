### 2.2.6 Define Updater

We use Backpropagation through time (BPTT) for optimize the RNNLM. BPTT can be implemented by overriding `update_core()` method of `StandardUpdater`. First, in the constructor of the `BPTTUpdater`, it takes `bprop_len` as an argument in addiotion to other arguments `StandardUpdater` needs. `bprop_len` defines the length of sequence $T$ to calculate the loss:

$$
\mathcal{L} = - \sum_{t=0}^T \sum_{n=1}^{|\mathcal{V}|}
\hat{P}({\bf x}_{t+1}^{(n)})
\log
P_{\rm model}({\bf x}_{t+1}^{(n)} \mid {\bf x}_t^{(n)})
$$

where $\hat{P}({\bf x}_t^n)$ is a probability for $n$-th word in the vocabulary at the position $t$ in the training data sequence.
