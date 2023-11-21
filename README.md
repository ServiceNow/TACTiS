# TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series

Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Nicolas Chapados, Alexandre Drouin (2023). *[TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series](https://arxiv.org/abs/2310.01327)*. (Preprint)

> We introduce a new model for multivariate probabilistic time series prediction, designed to flexibly address a range of tasks including forecasting, interpolation, and their combinations. Building on copula theory, we propose a simplified objective for the recently-introduced transformer-based attentional copulas (TACTiS), wherein the number of distributional parameters now scales linearly with the number of variables instead of factorially. The new objective requires the introduction of a training curriculum, which goes hand-in-hand with necessary changes to the original architecture. We show that the resulting model has significantly better training dynamics and achieves state-of-the-art performance across diverse real-world forecasting tasks, while maintaining the flexibility of prior work, such as seamless handling of unaligned and unevenly-sampled time series.

[[Preprint]](https://arxiv.org/abs/2310.01327)

<br />

<p align="center">
  <img src="https://github.com/ServiceNow/tactis/blob/tactis-2/cover.png?raw=true" width="500" />
</p>


## Installation

You can install the TACTiS-2 model with [pip](https://pip.pypa.io/):

```console
pip install tactis
```

Alternatively, the `research` version installs `gluonts` and `pytorchts` as dependencies which are required to replicate experiments from the paper:

```console
pip install tactis[research]
```

Note: `tactis` has been currently tested with Python 3.10.8.

## Details

With the `research` version of the code, [`train.py`](https://github.com/ServiceNow/tactis/blob/tactis-2/train.py) can be used to train the TACTiS-2 model for a specific dataset. The arguments in [`train.py`](https://github.com/ServiceNow/tactis/blob/tactis-2/train.py) can be used to specify the dataset, the training task (forecasting or interpolation), the hyperparameters of the model and a whole range of other training options.
More features such as interfaces to evaluate the model and obtain metrics, as well as demo notebooks will be added to this codebase soon.

For an implementation of the original version of TACTiS, please see [here](https://github.com/ServiceNow/tactis/tree/v1.0.0).

## Note

More features are coming soon!

## Citing this work

Please use the following Bibtex entry to cite TACTiS-2.

```
@misc{ashok2023tactis2,
      title={TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series}, 
      author={Arjun Ashok and Étienne Marcotte and Valentina Zantedeschi and Nicolas Chapados and Alexandre Drouin},
      year={2023},
      eprint={2310.01327},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
