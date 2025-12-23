# app/word_prediction.py
# ⚠️ NO IMPORTS ALLOWED IN THIS FILE

class WordPredictor:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add_word(self, word):
        if not word:
            return

        self.history.append(word)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def predict(self):
        if not self.history:
            return []

        # return last unique words as suggestions
        return list(dict.fromkeys(self.history[::-1]))[:3]
