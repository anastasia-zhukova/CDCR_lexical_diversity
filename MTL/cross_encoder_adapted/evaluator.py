class Evaluation:
    def __init__(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        self.tp = ((predictions == 1) & (labels == 1)).sum()
        self.fp = ((predictions == 1) & (labels != 1)).sum()
        self.fn = ((predictions != 1) & (labels == 1)).sum()
        self.tn = ((predictions != 1) & (labels != 1)).sum()
        self.total = len(labels)

        self.precision = self.tp / (self.tp + self.fp) if self.tp + self.fp != 0 else 0
        self.recall = self.tp / (self.tp + self.fn) if self.tp + self.fn != 0 else 0

    # Other methods remain the same

    def get_accuracy(self):
        return (self.tp + self.tn) / self.total

    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall

    def get_f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0


