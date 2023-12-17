import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Constants for abbreviation substitutions
ABBREVIATIONS = {
    "W/D": "withdraw",
    "_F": "",
    "_V": "", 
    "PTS": "transfer"
}

# Download resources for text processing
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TransactionCategorizer:
    """
    Class for categorizing bank transaction descriptions.
    """

    def __init__(self, model_name='facebook/bart-large-mnli'):
        """
        Initializes the classifier pipeline and loads keyword dictionaries.
        """
        self.classifier = pipeline('zero-shot-classification', model=model_name)
        self.keywords_income = self._load_keywords('keywords_dict_income.json')
        self.keywords_expenses = self._load_keywords('keywords_dict_expenses.json')

    @staticmethod
    def _load_keywords(file_path: str) -> dict:
        """
        Load keywords from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing keywords.

        Returns:
            dict: Dictionary containing categories and associated keywords.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def preprocess_text(text: str, use_stemming: bool = False) -> str:
        """
        Cleans and preprocesses the input text.

        Args:
            text (str): The raw text.
            use_stemming (bool): Use stemming if True, otherwise use lemmatization.

        Returns:
            str: The processed text.
        """
        # Substitute abbreviations
        for abb, full in ABBREVIATIONS.items():
            text = text.replace(abb, full)

        # Convert text to lowercase and remove URLs
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove numbers, special characters, and extra spaces
        text = re.sub(r'[\W0-9]+', ' ', text).strip()

        # Tokenization, stopwords removal, and stemming/lemmatization
        tokens = [word for word in text.split() if word not in stopwords.words('english')]
        if use_stemming:
            tokens = [nltk.PorterStemmer().stem(word) for word in tokens]
        else:
            tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]

        return ' '.join(tokens)

    def categorize_with_keywords(self, description: str, keywords_dict: dict) -> str:
        """
        Assigns a category based on keyword matching.

        Args:
            description (str): The transaction description.
            keywords_dict (dict): Dictionary of categories and keywords.

        Returns:
            str: The assigned category.
        """
        processed_desc = self.preprocess_text(description)
        for category, keywords in keywords_dict.items():
            if any(keyword in processed_desc for keyword in keywords):
                return category
        return "Uncategorized"

    def categorize_with_transformer(self, description: str) -> str:
        """
        Uses transformer model (BART) to categorize a description.

        Args:
            description (str): The transaction description.

        Returns:
            str: The assigned category.
        """
        processed_desc = self.preprocess_text(description)
        if not processed_desc:
            return 'Other Expenses'
        categories = list(self.keywords_expenses.keys())
        prediction = self.classifier(processed_desc, categories)
        return prediction['labels'][0]

    def categorize_transactions(self, input_file: str, output_file: str) -> None:
        """
        Processes an input CSV file, categorizes each transaction, and saves the results.

        Args:
            input_file (str): Path to the input CSV.
            output_file (str): Path to save the categorized output CSV.
        """
        df = pd.read_csv(input_file)

        # Separate income and expenses transactions
        income = df[df["Credit"].notna()].copy()
        expenses = df[df["Debit"].notna()].copy()

        # Assign categories using keywords
        income["Category"] = income["Description"].apply(lambda x: self.categorize_with_keywords(x, self.keywords_income))
        expenses["Category"] = expenses["Description"].apply(lambda x: self.categorize_with_keywords(x, self.keywords_expenses))

        # Merge income and expenses
        df_processed = pd.concat([income, expenses])

        # If any transaction is "Uncategorized", use the transformer to assign a category
        uncategorized_mask = df_processed["Category"] == "Uncategorized"
        if any(uncategorized_mask):
            df_processed.loc[uncategorized_mask, "Category"] = df_processed.loc[uncategorized_mask, "Description"].apply(self.categorize_with_transformer)

        # Save the categorized transactions to a CSV
        df_processed.to_csv(output_file, index=False)

# Usage example:
# categorizer = TransactionCategorizer()
# categorizer.categorize_transactions('path/to/input.csv', 'path/to/output.csv')
