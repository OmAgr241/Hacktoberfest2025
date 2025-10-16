import nltk
import re
import heapq
import os

# Download required nltk data only once
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

class SmartSummarizer:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(text)
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self):
        clean_text = re.sub(r'\s+', ' ', self.text)
        clean_text = re.sub(r'\[[0-9]*\]', ' ', clean_text)
        clean_text = re.sub(r'[^\w\s]', '', clean_text)
        return clean_text.lower()

    def summarize(self, max_sentences=3):
        clean_text = self.preprocess()
        words = word_tokenize(clean_text)
        freq_table = {}

        for word in words:
            if word not in self.stop_words:
                freq_table[word] = freq_table.get(word, 0) + 1

        max_freq = max(freq_table.values())
        for word in freq_table.keys():
            freq_table[word] /= max_freq

        sentence_scores = {}
        for sent in self.sentences:
            for word in word_tokenize(sent.lower()):
                if word in freq_table:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + freq_table[word]

        summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
        return ' '.join(summary_sentences)

def summarize_file(filepath, lines=3):
    if not os.path.exists(filepath):
        print("❌ File not found.")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    summarizer = SmartSummarizer(text)
    summary = summarizer.summarize(lines)
    print("\n🧩 === Summary ===\n")
    print(summary)
    print("\n✅ Summary generated successfully.")

if __name__ == "__main__":
    print("🧠 === AI Notes Summarizer ===")
    choice = input("Summarize from (1) Text or (2) File? Enter 1 or 2: ")

    if choice == '1':
        print("\nEnter your text (type 'END' on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        text = '\n'.join(lines)

        summarizer = SmartSummarizer(text)
        print("\n🧩 === Summary ===\n")
        print(summarizer.summarize())
        print("\n✅ Done!")
    elif choice == '2':
        filepath = input("Enter file path: ")
        summarize_file(filepath)
    else:
        print("❌ Invalid choice.")
