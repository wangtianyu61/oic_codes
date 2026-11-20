# Code for "Optimizer's Information Criterion: Dissecting and Correcting bias in Data-Driven Optimization"

> Garud Iyengar, Henry Lam, Tianyu Wang
>
> Columbia University.

This codebase corresponds to the [paper](https://arxiv.org/pdf/2306.10081).

If you find this repository useful in your research, please cite the following paper:

```bibtex
@article{iyengar2023optimizer,
  title={Optimizer's information criterion: Dissecting and correcting bias in data-driven optimization},
  author={Iyengar, Garud and Lam, Henry and Wang, Tianyu},
  journal={Management Science (forthcoming)},
  year={2026}
}
```

## Table of Contents

- [Code for "Optimizer's Information Criterion: Dissecting and Correcting bias in Data-Driven Optimization"](#code-for-optimizers-information-criterion-dissecting-and-correcting-bias-in-data-driven-optimization)
  - [Table of Contents](#table-of-contents)
  - [Computational Requirements](#computational-requirements)
  - [Repository Layout](#repository-layout)
  - [Data](#data)
    - [Source](#source)
  - [Reproducing Paper Results](#reproducing-paper-results)
    - [Main body](#main-body)
      - [Figures before the experimental parts](#figures-before-the-experimental-parts)
      - [Section 5.1](#section-51)
      - [Section 5.2](#section-52)
      - [Section 5.3](#section-53)
    - [Appendix](#appendix)
  - [Questions and Issues](#questions-and-issues)

## Computational Requirements

The code has been tested with Python 3.9.7 and depends on PyTorch, scikit-learn, and CVXPY. It uses NumPy, SciPy, and pandas for numerical computation and data manipulation, matplotlib and seaborn for figures, and tqdm for progress reporting. The following commands install the package versions in the environment used to check the current code:

```bash
python -m pip install \
  numpy==1.22.4 scipy==1.7.3 pandas==1.5.3 \
  scikit-learn==1.6.1 torch==2.2.2 cvxpy==1.6.0 \
  Mosek==10.2.19 matplotlib==3.5.0 seaborn==0.13.2 tqdm==4.67.1
```

The portfolio experiments explicitly use the MOSEK solver through CVXPY. A working MOSEK installation and license are therefore required; see the [MOSEK installation guide](https://docs.mosek.com/latest/pythonapi/install-interface.html) and [license setup instructions](https://docs.mosek.com/latest/licensing/quickstart.html). 

## Repository Layout
The `src` folder include source codes of the three setups in Section 5. 
- `portfolio_model.py`: bias computation of the portfolio allocation problem in Section 5.1;
- `nv_model.py`: bias computation of the single-item newsvendor in Section 5.2;
-  `models.py`: basic ml models in the regression in Section 5.3.

Experiment scripts and notebooks use the shared paths in `src/paths.py`. Generated CSV and NumPy files are saved under `output/`, and generated figures are saved under `output/figures/`; parent directories are created automatically. Existing saved input files remain under their original `output/` subdirectories. Paths are resolved from the repository location rather than the current working directory. The notebooks can be opened from the repository root or `case_study/`.

```text
output/nv/                  Newsvendor experiment outputs.
output/portfolio/           Portfolio experiment outputs.
output/port_selection/      Portfolio selection outputs.
output/figures/             Generated figures.
output/figures/portfolio/   Portfolio case-study figures.
output/figures/nv2/         Newsvendor comparison figures.
```

## Data

### Source

We consider three different set of tasks, (1) portfolio optimization; (2) newsvendor; and the regression tasks. The portfolio optimization and newsvendor tasks are based on the synthetic data, and the python files include their data-generating processes.

For the regression task, we use the following data surce:
- Wine Quality: [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) (CC BY 4.0).

The regression experiment uses the UCI Wine Quality dataset of red and white Portuguese Vinho Verde wines ([Cortez et al., 2009](https://archive.ics.uci.edu/dataset/186/wine+quality); [Duchi et al., 2023](https://doi.org/10.1287/opre.2022.2363)). The downloaded files contain $N = 6,497$ observations: 1,599 red wines and 4,898 white wines. Each wine is associated with a feature vector $z \in \mathbb{R}^{12}$ comprising 11 physicochemical measurements and a wine-type indicator, and a label $\xi \in \mathbb{R}$ representing a subjective quality assessment.

The data files have already been downloaded to the following paths, relative to the repository root:

```text
src/datasets/wine/winequality-red.csv
src/datasets/wine/winequality-white.csv
```

The loader in `src/datasets/tabular.py` combines the two files, adds a red/white wine indicator, and uses the `quality` column as the regression label. For reuse and attribution requirements, see the [UCI dataset page](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Reproducing Paper Results

We provide the specific reproducing guideline of all figures, tables and results in the final paper. Some of the sampled temporary outputs are described in 

### Main body

#### Figures before the experimental parts
- Figure 1: the same figure as Figure 3(a), see "Section 5.3" as follows for more details.
- Figure 2: case_study/visual_flow.ipynb


#### Section 5.1
For replicable use of the portfolio allocation experiment:
- Table 3: visual_ops.ipynb
- Figure 3 (a): generated from `case_study/dro_portfolio_visual.ipynb` with sampled output, `case_study/dro_portfolio.py` corresponds to the decision selection for the DRO model, with a more user-friendly extension in the dro package `dro` with detailed descriptions in https://python-dro.org/api/notebooks/f_dro_tutorial.html.
- Figure 3 (b): selection.ipynb

Here, `portfolio.py` calculates the evaluation bias for each evaluation procedure;
  

#### Section 5.2

For replicable use of the newsvendor experiment:
- Figure 4 (a)(b)(c): visual_ops.ipynb
- Figure 4 (d): selection.ipynb

Here, `newsvendor.py` calculates the bias of each decision method for each evaluation procedure, including CV and bootstrap methods.

#### Section 5.3

For replicable use of the regression experiment:

- Table 4: `regression.py`, without any saved output.
  

### Appendix
- Tables EC.1 - EC.3: same as the way generated in Table 3 from `visual_ops.ipynb`.
- Figure EC.1: vary the parameter choices of `case_study/dro_portfolio.ipynb` and `case_study/dro_portfolio.py`.
- Figure EC.2:  visual_ops.ipynb
- Table EC.4: case_study/risk_portfolio.py
- Figures EC.3 - EC.4: visual_ops.ipynb

P.S. For those who are interested in the uncertainty quantification of performance evaluation in general (nonparametric) data-driven decision-making problems, please refer to the [paper](https://openreview.net/pdf?id=4lGPSbGe11) and the corresponding [codebase](https://github.com/wangtianyu61/CV_GoldStandard).

## Questions and Issues

For questions about reproducing the benchmark or problems with the code, please open an issue at this repo or contact tw2837@columbia.edu.

