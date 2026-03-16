\# NLP Lab 5 — Dataset Splitting and Leakage Checks



\## Project Overview



This project implements dataset splitting and leakage detection procedures for a binary text classification task: \*\*Ukrainian news classification (Real vs Fake)\*\*.



The goal of this laboratory work is to ensure that the dataset is properly prepared for machine learning experiments and that no \*\*data leakage\*\* exists between training, validation, and test sets.



The dataset consists of Ukrainian news texts that have been preprocessed during previous laboratory work (Lab 2).



---



\## Dataset



The dataset contains the following fields:



| Column         | Description                      |

| -------------- | -------------------------------- |

| processed\_text | preprocessed Ukrainian news text |

| label          | news label (Real / Fake)         |



During the experiment an additional column is generated:



| Column  | Description                             |

| ------- | --------------------------------------- |

| text\_id | unique identifier assigned to each text |



---



\## Dataset Splitting Strategy



A \*\*stratified random split\*\* is used to preserve the class distribution across dataset partitions.



Split proportions:



\* \*\*Train:\*\* 80%

\* \*\*Validation:\*\* 10%

\* \*\*Test:\*\* 10%



Random seed used for reproducibility:



```

seed = 42

```



Stratification is performed using the label column.



---



\## Leakage Checks



Several types of potential dataset leakage are evaluated.



\### 1. Exact Duplicate Detection



Texts are compared across splits to ensure that identical documents do not appear in multiple subsets.



Intersections checked:



\* Train ∩ Validation

\* Train ∩ Test

\* Validation ∩ Test



---



\### 2. Near-Duplicate Detection



Near-duplicate texts are detected using:



\* \*\*TF-IDF vectorization\*\*

\* \*\*Cosine similarity\*\*



Pairs with similarity greater than \*\*0.95\*\* are flagged as suspicious.



---



\### 3. Template Leakage Detection



The dataset is scanned for patterns that may reveal labels directly in the text.



Examples of patterns checked:



```

label=

class=

category=

fake

real

```



These patterns may indicate potential label leakage.



---



\## Generated Artifacts



Running the notebook produces the following files.



\### Dataset Split Files



```

splits\_train\_ids.txt

splits\_val\_ids.txt

splits\_test\_ids.txt

```



These files contain the `text\_id` values for each dataset split.



---



\### Manifest File



```

splits\_manifest\_lab5.json

```



Contains metadata about the split:



\* split strategy

\* seed

\* dataset sizes

\* label distribution



---



\### Leakage Report



```

leakage\_risk\_report\_lab5.md

```



Summarizes the results of leakage checks.



---



\### Audit Summary



```

audit\_summary\_lab5.md

```



Contains a concise overview of dataset statistics and leakage analysis.



---



\## Installation



Install required Python packages:



```

pip install -r requirements.txt

```



---



\## Running the Notebook



Open and run the notebook:



```

notebooks/lab5\_split\_leakage\_checks.ipynb

```



The notebook will:



1\. Load the processed dataset

2\. Generate dataset splits

3\. Perform leakage checks

4\. Produce audit and report files



---



\## Reproducibility



The experiment uses a fixed random seed:



```

42

```



This ensures that dataset splits remain consistent across runs.



---



\## Author



Uliana — NLP coursework project



