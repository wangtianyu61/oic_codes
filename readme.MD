This codebase corresponds to the <a href = "https://arxiv.org/pdf/2306.10081">paper</a>.

Source codes of the three setups in Section 5 are demonstrated in ``src`` folder. The bias computation of the single-item newsvendor is in ``nv_model.py``; the bias computation of the portfolio allocation problem is in ``portfolio_model.py``; the basic models in the ml regression are ``models.py``.


For replicable use of the portfolio allocation experiment:
- ``bias_portfolio.py`` calculates the bias for each decision method;
- ``saa_portfolio.py`` corresponds to the bias for the SAA method; 
- ``dro_portfolio.py`` corresponds to the decision selection for the DRO model, with a more user-friendly extension in the dro package ``dro`` with detailed descriptions in https://python-dro.org/api/notebooks/f_dro_tutorial.html).
- ``selection.ipynb`` (first part) selects the best distribution class in the estimate-then-optimize approach.



For replicable use of the newsvendor experiment:
- ``newsvendor.py`` calculates the bias of each decision method for each evaluation procedure, including CV and bootstrap methods. 
- ``selection.ipynb`` (second part) selects the best constrained size in the portfolio optimization approach.



For replicable use of the regression experiment:
- ``wine.py`` computes the bias of each decision method, ``debias_wine_AOCV()`` using LOO method and ``debias_wine()`` using ALO method.



If you find this repository useful in your research, please cite the following paper:

```
@article{iyengar2023optimizer,
  title={Optimizer's information criterion: Dissecting and correcting bias in data-driven optimization},
  author={Iyengar, Garud and Lam, Henry and Wang, Tianyu},
  journal={arXiv preprint arXiv:2306.10081},
  year={2023}
}
```

If you are interested in the performance evaluation in general nonparametric decision models and the uncertainty quantification problem, please refer to the <a href = "https://openreview.net/pdf?id=4lGPSbGe11" >paper</a> and the corresponding <a href = "https://github.com/wangtianyu61/CV_GoldStandard">codebase</a>.
