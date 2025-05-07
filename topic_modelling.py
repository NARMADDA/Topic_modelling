import re
import time
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel

# Setup Selenium
options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

# Open Technology section
driver.get("https://indianexpress.com/section/technology/")
time.sleep(3)

# Extract article links
articles = []
elements = driver.find_elements(By.TAG_NAME, 'h2')
for el in elements:
    try:
        a_tag = el.find_element(By.TAG_NAME, 'a')
        title = a_tag.text.strip()
        url = a_tag.get_attribute('href')
        articles.append((title, url))
    except:
        continue

# Fetch full text from first 5 articles
def get_full_article(url):
    driver.get(url)
    time.sleep(2)
    try:
        paras = driver.find_elements(By.CSS_SELECTOR, 'div.articles p')
        text = ' '.join(p.text for p in paras if p.text)
        print(f"Extracted text: {text[:300]}")  # Print a snippet
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

texts = [get_full_article(link) for _, link in articles[:5]]

# Print preview of texts
for i, text in enumerate(texts):
    print(f"\n--- Article {i+1} Preview ---\n{text[:300]}")

driver.quit()

# Text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())
    words = word_tokenize(text)
    return [word for word in words if word not in stop_words and len(word) > 3]

processed_texts = [clean_text(text) for text in texts]

# Print tokenized texts for debugging
for i, tokens in enumerate(processed_texts):
    print(f"\n--- Tokens from Article {i+1} ---\n{tokens[:10]}")  # Print first 10 tokens

# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Train LDA
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=10)

# Show topics
topics = lda_model.print_topics()
for i, topic in topics:
    print(f"\nTopic {i+1}: {topic}")
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_indianexpress.html')