"""Labelled data evaluation module."""
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from plots import visualisations


def labelled_data_evaluation(true_labels, preds, preds_score, model_name):
    """Function to perform evaluation of the binary classifiers via the use of some standard metrics in the process."""
    try:
        assert len(true_labels) == len(preds) == len(preds_score), "Testing set sizes are different."

        tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
        n_samples = len(true_labels)
        auc = roc_auc_score(true_labels, preds_score[:, 1])

        print("\n".join([
            f"___",
            f"Results for {model_name}",

            f"Total number of sample: {n_samples}",
            f"Total number of positive samples: {tp + fn} {(tp + fn) / n_samples:.2f}",
            f"Total number of negative samples: {tn + fp} {(tn + fp) / n_samples:.2f}",

            f"TN {tn} ({tn / n_samples:.2f})",
            f"FP {fp} ({fp / n_samples:.2f})",
            f"FN {fn} ({fn / n_samples:.2f})",
            f"TP {tp} ({tp / n_samples:.2f})",

            f"TPR, Sensitivity {tp / (tp + fn):.2f}",  # True Positive Rate
            f"FNR, Miss Rate {tn / (tp + fn):.2f}",  # False Negative Rate
            f"TNR, Specificity {tn / (tn + fp):.2f}",  # True Negative Rate
            f"FPR, Fall-out {fp / (tn + fp):.2f}",  # False Positive Rate

            f"Positive Predictive Value, Precision {tp / (tp + fp):.2f}",
            f"False Discovery Rate {fp / (tp + fp):.2f}",
            f"Negative Predictive Value {tn / (tn + fn):.2f}",
            f"False Omission Rate {fn / (tn + fn):.2f}",

            f"ROC AUC {auc:.2f}",
            f"___",

        ]))

        fpr, tpr, thresholds = roc_curve(true_labels, preds_score[:, 1])
        visualisations.plot_roc_auc_curve(fpr, tpr, auc, model_name=model_name)

    except Exception or AttributeError as e:
        print(e)
