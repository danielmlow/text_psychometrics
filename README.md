# zero_shot

Code for: Low DM, Mair P, Nock MK, Ghosh SS. (2025). Text psychometrics: assessing psychological constructs in text. PsyArxiv.

# Code 

```
conda create -y -n text_psychometrics python=3.10 seaborn ipywidgets ipykernel
conda activate text_psychometrics
```

Glossary:
- ctl: Crisis Text Line data (de-identified sensitive data, only run locally)
- reddit: data from 13 public mental health subreddits (4600 posts each)
- srl: Suicide Risk Lexicon

## Preprocessing
- srl_constructs.py 

CTL
- ctl_clean_datasets.py
- ctl_train_test_split.ipynb
- ctl_create_df.ipynb 
- ctl_feature_extraction.py

Reddit
- reddit_download.ipynb Download Reddit dataset. 
- reddit_feature_extraction_srl_liwc_cts.ipynb for Suicide Risk Lexicon, LIWC, CTS


## Results
- Table 2: Performance of different types of text classification methods. 
    Replace <dataset> with either `ctl` for Crisis Text Line or `reddit` for Reddit dataset
    - LIWC, SRL (Suicide Risk Lexicon) using log. reg. 
        - classify_<dataset>.py # srl and LIWC
        - classify_<dataset>_construct_text_similarity.py 
    - LLMs: run on cloud API for Reddit data
        - - classify_reddit_llms.ipynb 
    - LLMs: run on local GPUs for private CTL data
        - prompts.py
        - open_source_genai.sh
        - open_source_genai_gemma2.sh
        - open_source_genai_gemma7.sh
        - open_source_genai.ipynb
        - open_source_genai.py
        - open_source_genai_results.py
        - Content validity
            - open_source_genai_content_validity.ipynb
            - open_source_genai_content_validity.py
    - classify_<dataset>_results.py taking results from above and creating tables for paper
    - Full results: results_25-03-18T17-25-24results_content_validity_25-03-18T17-25-24_full_content_validity_all.csv

- Figure 6 (Predicting a request for an emergency services intervention) & Table 3 (Feature importance analysis)
    - desire_vs_imminent_risk_cts_feature_extraction.ipynb preprocessing and CTS feature extraction
    - desire_vs_imminent_risk.ipynb models
    - desire_vs_imminent_risk_over_time.py