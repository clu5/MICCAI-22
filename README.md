# Improving Trustworthiness of AI Disease Severity Rating in Medical Imaging with Ordinal Conformal Prediction Sets

## Abstract
The regulatory approval and broad clinical deployment of medical AI have been hampered by the perception that deep learning models fail in unpredictable and possibly catastrophic ways. A lack of statistically rigorous uncertainty quantification is a significant factor undermining trust in AI results. Recent developments in distribution-free uncertainty quantification present practical solutions for these issues by providing reliability guarantees for black-box models on arbitrary data distributions as formally valid finite-sample prediction intervals. Our work applies these new uncertainty quantification methods -- specifically conformal prediction -- to a deep-learning model for grading the severity of spinal stenosis in lumbar spine MRI. We demonstrate a technique for forming ordinal prediction sets that are guaranteed to contain the correct stenosis severity within a user-defined probability (confidence interval). On a dataset of 409 MRI exams processed by the deep-learning model, the conformal method provides tight coverage with small prediction set sizes. Furthermore, we explore the potential clinical applicability of flagging cases with high uncertainty predictions (large prediction sets) by quantifying an increase in the prevalence of significant imaging abnormalities (e.g. motion artifacts, metallic artifacts, and tumors) that could degrade confidence in predictive performance when compared to a random sample of cases.

* [Paper](https://arxiv.org/abs/2207.02238)
* Presented @ [MICCAI 2022](https://conferences.miccai.org/2022/en/)!


```
@misc{https://doi.org/10.48550/arxiv.2207.02238,
  doi = {10.48550/ARXIV.2207.02238},
  url = {https://arxiv.org/abs/2207.02238},
  author = {Lu, Charles and Angelopoulos, Anastasios N. and Pomerantz, Stuart},
  keywords = {Machine Learning (cs.LG), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  title = {Improving Trustworthiness of AI Disease Severity Rating in Medical Imaging with Ordinal Conformal Prediction Sets},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
