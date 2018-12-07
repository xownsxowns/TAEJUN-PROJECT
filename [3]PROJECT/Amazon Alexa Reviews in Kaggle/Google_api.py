from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import pandas as pd

review_data = pd.read_csv('amazon_alexa.tsv', delimiter='\t', encoding='utf-8')

path = 'C:\\Users\jhpark\Documents\GitHub\Private-NLP\Taejun-1fa51108e2e7.json'  # FULL path to your service account key
client = language.LanguageServiceClient.from_service_account_json(path)

# The text to analyze
text = 'Hello, world'
document = types.Document(
    content = text,
    type = enums.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment = client.analyze_sentiment(document=document).document_sentiment

print('Text: {}'.format(text))
print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))