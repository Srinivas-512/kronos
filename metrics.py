import sklearn.metrics as metrics
import warnings

def classification_scores(y_true, y_pred, y_relax, suppress_warning=True ):
    scores = {}

    with warnings.catch_warnings():
        if suppress_warning:
            warnings.simplefilter("ignore")
        scores['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        # Average of recalls obtained for each class
        scores['balanced_accuracy'] = metrics.balanced_accuracy_score(y_true, y_pred)
        scores['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)
        # Suppress 'UserWarning: y_pred contains classes not in y_true' in console output
        scores['classification_report'] = metrics.classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        results = []
        for pred, correct_set in zip(y_true, y_relax):
            if pred in correct_set:
                results.append(1)
            else:
                results.append(0)
        
        results_correct = [1]*len(y_true)
        scores['relaxed_accuracy'] = metrics.accuracy_score(results_correct, results)
    
    return scores
