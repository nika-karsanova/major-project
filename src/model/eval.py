from sklearn.metrics import confusion_matrix


def labelled_data_evaluation(true_labels, preds):
    """Function to perform evaluation of the binary classifiers via the use of some standard metrics in the process."""
    tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
    n_samples = len(true_labels)

    print("\n".join([
        f"TN {tn} ({tn / n_samples:.2f})",
        f"FP {fp} ({fp / n_samples:.2f})",
        f"FN {fn} ({fn / n_samples:.2f})",
        f"TP {tp} ({tp / n_samples:.2f})",

        f"TPR, Sensitivity {tp / (tp + fn):.2f}"  # True Positive Rate
        f"FNR {tn / (tp + fn):.2f}"  # False Negative Rate
        f"TNR, Specificity {tn / (tn + fp):.2f}"  # True Negative Rate
        f"FPR {fp / (tn + fp):.2f}"  # False Positive Rate
        
        f"Positive Predictive Value, Precision {tp / (tp + fp):.2f}",
        f"False Discovery Rate {fp / (tp + fp):.2f}",
        f"Negative Predictive Value {tn / (tn + fn):.2f}",
        f"False omission Rate {fn / (tn + fn):.2f}",

        f"AUC ",
        f"ROC ",

    ]))
