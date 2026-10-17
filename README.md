# Telecommunications Customer Churn Prediction — Research Practice

**Assignment 2 submission package** | Shahriar Kamal | October 2026

This repository examines customer churn in the 7,043-record IBM Telco Customer Churn sample dataset. It tests class-weighted Logistic Regression against Random Forest, reporting churn-class F1, precision, recall and ROC-AUC.

## Submission files

| Deliverable | Link | Description |
| --- | --- | --- |
| Research paper | **[paper.md](paper.md)** | Abstract, literature-backed introduction, methods, results with figures, discussion, conclusion and APA references |
| Experiment notebook | **[experiment.ipynb](experiment.ipynb)** | Original experiment with saved model results, ROC curves and feature importance |
| Slide deck (6 slides) | **[slides/churn_research_presentation.pptx](slides/churn_research_presentation.pptx)** | Problem, literature, methodology, results and takeaway |
| Original literature review | [literature_review.md](literature_review.md) | Assignment 1 literature review |
| Original methodology | [methodology.md](methodology.md) | Assignment 1 research plan |
| Figures | [figures/](figures/) | Matplotlib plots and the original notebook's comparison plot |
| Figure source script | [generate_figures.py](generate_figures.py) | Reproduces Figures 1 and 2 from the notebook's printed test results |
| Evaluation values | [results.json](results.json) | Scores and confusion-matrix counts |
| Zotero import | [references.ris](references.ris) | Five verified journal references in RIS format, ready for import |

## Main results

| Model | Churn F1 | ROC-AUC | Churn precision | Churn recall |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6155 | 0.8461 | 0.5069 | **0.7834** |
| Random Forest | **0.6261** | **0.8471** | **0.5275** | 0.7701 |

The ensemble improves F1 by just 0.0106; it reduces false positives from 285 to 258 while increasing false negatives from 81 to 86. These are modest improvements in one hold-out split, not proof of universally superior performance.

## Reproducing the figures

```bash
pip install -r requirements.txt
python generate_figures.py
```

`generate_figures.py` reads `results.json` and regenerates the bar chart and confusion matrices. These numbers originate in the saved `experiment.ipynb` output; this script does **not** retrain the models. Figure 3 is the original saved notebook plot.

## Re-running the modeling notebook

1. Download the IBM Telco Customer Churn CSV named `WA_Fn-UseC_-Telco-Customer-Churn.csv`, e.g. from [Kaggle's Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Open `experiment.ipynb` in Google Colab and upload the CSV to `/content/`, as expected by its original code, or modify the `pd.read_csv(...)` path to your local file.
3. Run cells in order. The notebook uses `random_state=42`, `test_size=0.20`, and stratification. It reports Random Forest and Logistic Regression performance plus ROC and feature-importance plots.

**Important:** The CSV is not bundled in this repository package because it was not supplied. The experimental scores are from the saved notebook output, not newly verified training runs. The IBM dataset is illustrative sample data, not a live production customer record. The original notebook uses its own preprocessing logic as submitted.

## Reference management

All academic references in `paper.md` use APA 7 formatting. To manage them in Zotero, open **File → Import → A file** and choose `references.ris`; select **American Psychological Association 7th edition** when generating a bibliography. These five journal articles are traceable by DOI, and can be searched in [Google Scholar](https://scholar.google.com/). An IBM dataset documentation reference is listed separately in the paper.

## Submission

Upload all files to your course GitHub repository (the branch must contain these files). Submit the URL for that branch or repository once you can see `paper.md`, the `.pptx`, and `README.md` on GitHub. **Files prepared locally are not yet pushed.**
