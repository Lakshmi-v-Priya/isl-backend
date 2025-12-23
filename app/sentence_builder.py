class SentenceBuilder:
    def __init__(self):
        self.words = []

    def add_word(self, word):
        # Remove continuous duplicates
        if not self.words or self.words[-1] != word:
            self.words.append(word)

    def clear(self):
        self.words = []

    def build_sentence(self):
        # Simple ISL → English reordering rules
        # Example: YOU NAME WHAT → What is your name
        if self.words == ["YOU", "NAME", "WHAT"]:
            return "What is your name"

        return " ".join(self.words).capitalize()
