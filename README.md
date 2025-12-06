# metagenomics_sample_classification
Implementation an algorithm that will generate sketches of the provided large metagenomics samples from various environments, and then use the resulting sketches to classify new, smaller samples from these environments



# Algorithms for Genomic Data Analysis  
## Assignment 2: Metagenomics Sample Classification  
**Winter semester 2025/2026**

---

## **Task**

Implement an algorithm that will generate **sketches** of the provided large metagenomics samples from various environments, and then use the resulting sketches to **classify new, smaller samples** from these environments.

The training and testing data originate from metagenomic sequencing done by the **MetaSUB consortium**.  
Intended application: **forensics** — classifying samples from objects used in criminal activities.

Your program may use:

- code fragments from classes  
- Python standard library  
- NumPy, SciPy, Biopython  
- mmh3 (hashing)

Your program **cannot** use:

- programs/libraries to manipulate k-mer profiles, minimizers, sketching, etc.  
- multiprocessing  
- foreign language subprograms  
- JIT compilers

Your solution must include:

- `classifier.py` (Python 3)  
- slides describing your approach  

Avoid submitting packaged virtual environments.

---

## **Specification**

Program must be executable as:
python3 classifier.py training_data.tsv testing_data.tsv output.tsv


### **Training file format**

- TSV with ≥2 columns  
- First column: **gzipped FASTA filename**  
- Second column: **dataset class** (city)  
- Remaining columns can be ignored  
- Multiple datasets may share the same class  

### **Testing file format**

- Same format, but **only first column matters**  
- Contains gzipped FASTA filenames for testing datasets  

### **Data size**

- Reads are ~150 bp  
- Training: **20 datasets × 1M reads**  
- Testing: **50 datasets × 100k reads**  
- Extended testing: +20 training datasets and +100 test datasets  

### **Output format**

A tab-separated file:

- First column: dataset name  
- Next columns: classification scores for each class  
- Scores do **not** need to be probabilities  
- Larger value → higher likelihood  

---

## **Evaluation (AUC-ROC)**

For each class:

1. Compare returned scores with ground truth (0/1)  
2. Compute sensitivity & specificity for all thresholds  
3. Compute **ROC curve**  
4. Compute **AUC**  
5. Average AUC over all classes  

### **Minimum performance**

- **AUC-ROC ≥ 0.6**  
- Runtime ≤ **5 hours** on students' server  
- Memory ≤ **1 GB**

---

## **Additional Items (optional)**

For 2-person team: implement ≥1  
For 3-person team: ≥2

Each item requires **resampling** and plotting **mean ± SD**.

- Compare performance when **downsampling training data** (10–100%)  
- Compare performance when **downsampling testing data** (10–100%)  
- Evaluate performance on various **100k-read subsets** of large datasets  
  - Available via SRA: `SRRnnnnnnnn`  

---

## **Attached Files**

- Example input/output TSV files  
- Small training/testing FASTA files (downsampled to 1k sequences)  
- Ground truth for testing files  
- Program to evaluate classification  
- Full-size FASTA files on students server:  
  `/home/staff/iinf/ajank/adg`

---

## **Terms and Conditions**

Can be completed:

- Individually  
- In 2-person teams  
- In 3-person teams  

### **Schedule**

- Submit team: **Nov 31**  -> ***********  
- Submit code: **Dec 14** (Moodle)  
- Submit slides: **Dec 15**  
- Present in class: **Dec 16**

---

## **Assessment**

Minimum requirements → **2 points**

Extra points:

### **Classification performance (provided data)**  
- 2 pts: AUC-ROC ≥ 0.75  
- 1 pt : AUC-ROC ≥ 0.7  

### **Classification performance (extended data)**  
- 4 pts: ≥ 0.75  
- 3 pts: ≥ 0.7  
- 2 pts: ≥ 0.65  
- 1 pt : ≥ 0.6  

### **Runtime**  
- 3 pts: ≤ 1 min per 1M reads  
- 2 pts: ≤ 2 min  
- 1 pt:  ≤ 5 min  

### **Deadlines & presentation**  
- Up to 2 pts  

### **Additional items**  
- 1 pt per item  

### **Team size**  
- 2 pts: 1-person team  
- 1 pt : 2-person team  

**Maximum total: 15 points.**

---
