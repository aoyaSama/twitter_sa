import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

""" Plotting Module for Twitter Sentiment Analysis
"""

class Evaluator:
    """Class decorator for Evaluator .
    """

    def __init__(self, model, target_label):
        """Initialize self . data

        Args:
            model (str): ML Model name
            target_label ([str]): Label to evaluate against
        """
        self.model = model
        self.lb = target_label

        self.ac_scores = []
        self.f1_scores = []
        self.ngrams = []
        self.pp_method = []


    def add_prediction(self, prediction, pp_method, ngram='unigram'):
        """Add a prediction to the evaluator and print the Accuracy and F1 score .

        Args:
            prediction
            pp_method (str): Pre-process method
            ngram (str, optional): [ngram setting]. Defaults to 'unigram'.
        """
        ac = accuracy_score(self.lb, prediction)
        f1 = f1_score(self.lb, prediction, average="macro")

        self.ac_scores.append(ac)
        self.f1_scores.append(f1)
        self.pp_method.append(pp_method)
        self.ngrams.append(ngram)

        print("Pre-process Method: "+pp_method+f"\t\tAccuracy Score: " +
            f"{round(ac, 4)}\tMacro F1: {round(f1,4)}")


    def plot(self, score):
        """Make a bar plot of the model .

        Args:
            score (str): Option of ac = Accuracy Score or F1 = Major F1
        """

        score_types = ['ac', 'f1']
        if score not in score_types:
            raise ValueError("Invalid score type. Expected one of: %s" % score_types)

        if (score == 'ac'):
            y_label = "Accuracy score"
            scores = self.ac_scores

        if(score == 'f1'):
            y_label = "F1 Score"
            scores = self.f1_scores

        score_data = pd.DataFrame({'ngram': self.ngrams,
                      'pp_method': self.pp_method,
                      'score': scores
                      })

        plt = sns.catplot(x='ngram', y='score', data=score_data,
                  kind = "bar", hue='pp_method', legend=True)
        plt.set_axis_labels("ngram setting", y_label)
        plt.ax.set_title(self.model+" "+y_label+" between different Preprocess Methods")
        plt.set(ylim=(0.6,None))

        # plots the score value above each bar
        for c in plt.ax.containers:
            labels = [round(v.get_height(),3) for v in c]
            plt.ax.bar_label(c, labels=labels, padding=2)


    """Setters"""
    def set_model(self, model):
        """Set model .

        Args:
            model (str): ML Model name
        """

        self.model = model