# Categorization of Bank Statement Transactions

## 1. Overview
This document details the classification of bank statement descriptions into predefined categories using a hybrid method combining rule-based keyword matching and zero-shot text classification.

## 2. Dataset Preparation
The dataset was manually labeled to develop the classification system, with a focus on understanding distribution and identifying unique keywords for each category.

## 3. Categorization Approach
The process involves:
- **Data Segregation**: Income and expense transactions are treated separately.
- **Rule-Based Keyword Matching**: Utilizing a keyword dictionary for each category, with preprocessing steps such as text standardization and tokenization.
- **Zero-Shot Classification**: Employing a model for categorizing transactions not covered by rule-based methods.

## 4. Value and Expected Results
- **Measuring Success**: Using balanced accuracy as the primary metric.
- **Business Value**: Enhancing credit risk assessment, budgeting, financial planning, and trend analysis through accurate transaction categorization.

## 5. Future Work
Plans include expanding the labeled dataset, employing transfer learning with larger financial datasets, and considering deployment strategies for efficient model performance.
