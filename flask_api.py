from flask import Flask, request, jsonify
from categorize_transactions import TransactionCategorizer 

app = Flask(__name__)

# Initialize the TransactionCategorizer when the server starts.
categorizer = TransactionCategorizer()

@app.route('/categorize', methods=['POST'])
def categorize_transactions():
    """
    Categorize transactions based on descriptions and credit/debit information.
    This endpoint expects a JSON with a list of transactions. Each transaction should include 'description', 'credit', and 'debit' fields.
    """
    # Parse the incoming JSON data
    data = request.json

    if not data or 'transactions' not in data:
        return jsonify({'error': 'No transactions provided'}), 400

    try:
        transactions = data['transactions']

        # Prepare an empty list to hold the responses.
        categorized_transactions = []

        for transaction in transactions:
            description = transaction.get('description')
            credit = transaction.get('credit')  # This can be None if it's a debit entry
            debit = transaction.get('debit')  # This can be None if it's a credit entry

            # Validate if necessary fields are available.
            if description is None:
                return jsonify({'error': 'Missing transaction description'}), 400

            # Decide whether this is an income or an expense based on the presence of credit or debit info.
            if credit:
                category = categorizer.categorize_with_keywords(description, categorizer.keywords_income)
            elif debit:
                category = categorizer.categorize_with_keywords(description, categorizer.keywords_expenses)
            else:
                # Handle cases where both credit and debit are missing.
                return jsonify({'error': 'Missing credit and debit information'}), 400

            # Check if the transaction was not categorized. If so, use the transformer model.
            if category == "Uncategorized":
                category = categorizer.categorize_with_transformer(description)

            # Append the result to our list.
            categorized_transactions.append({
                'description': description,
                'credit': credit,
                'debit': debit,
                'category': category
            })

        # Return the list of categorized transactions.
        return jsonify(categorized_transactions)

    except Exception as e:
        # Handle any other kind of unexpected error.
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask web server.
    app.run(debug=True)
