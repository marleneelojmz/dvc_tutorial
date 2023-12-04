""" This script help us to evaluate
    the trained model.
    It's the source code to evaluate stage on dvc pipeline
"""
import sys
import json
import math
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from dvclive import Live

dvclive = Live()

if len(sys.argv) != 5:  
    # Lets validate the correct usage of the script,
    # the arguments will be given on the pipeline
    """ Arguments description:
            1.  Name of python script : evaluate.py
            2.  Name of model file : model.h5
            3.  output 1 : scores.json
            4.  output 2 : prc.json
            5.  output 3 : roc.json """

    sys.stderr.write("Arguments error. Usage: \n")
    sys.stderr.write("\t python evaluate.py model scores prc roc \n")
    sys.exit(1)

model_file = sys.argv[1]
scores_file = sys.argv[2]
prc_file = sys.argv[3]
roc_file = sys.argv[4]
image_size = [150]

model_saved = load_model(model_file)

test_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

test_generator = test_data_generator.flow_from_directory(
    "data/test/", target_size=(image_size[0], image_size[0]), shuffle=False
)
steps_per_epoch_test = len(test_generator)

y_true = test_generator.labels

test_history = model_saved.evaluate(test_generator, verbose=1)

y_pred = model_saved.predict(test_generator)

y_pred_proba = y_pred[:, 1]

precision, recall, prc_thresholds = metrics.precision_recall_curve(
    y_true, y_pred_proba
    )

fpr, tpr, roc_thresholds = metrics.roc_curve(y_true, y_pred_proba)

avg_prec = metrics.average_precision_score(y_true, y_pred_proba)
roc_auc = metrics.roc_auc_score(y_true, y_pred_proba)

# Saving the global metrics obtained for the actual model
with open(scores_file, "w") as fd:
    json.dump(
        {
            "avg_prec": avg_prec,
            "roc_auc": roc_auc,
            "accuracy": test_history[1],
            "loss": test_history[0],
        },
        fd,
        indent=4,
    )

dvclive.log_metric("avg_prec", avg_prec)
dvclive.log_metric("roc_auc", roc_auc)
dvclive.log_metric("accuracy", test_history[1])
dvclive.log_metric("loss", test_history[0])

# Saving ROC values obtained for the evaluated model
# ROC has a drop_intermediate arg that reduces the number of points
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# PRC lacks this arg, so we manually
# reduce to 1000 points as a rough estimate.

nth_point = math.ceil(len(prc_thresholds) / 1000)
prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

print(prc_points)

datapoints = [{'precision' : i_precision,'recall' : i_recall} for i_precision,i_recall in zip(precision, recall)]


with open(prc_file, "w") as fd:
    json.dump(
        {
            "prc": [
                {"precision": p.item(), "recall": r.item(),
                    "threshold": t.item()}
                for p, r, t in prc_points
            ]
        },
        fd,
        indent=4,
    )

dvclive.log_plot('Precision_Recall_Metric',
                 datapoints,
                 x = 'precision',
                 y = 'recall',
                 template = 'linear',
                 title = 'Precision / Recall metric thorught Threshold modif.'
                 )

with open(roc_file, "w") as fd:
    json.dump(
        {
            "roc": [
                {"fpr": fp.item(), "tpr": tp.item(), "threshold": t.item()}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        },
        fd,
        indent=4,
    )

datapoints = [{'FPR' : i_fpr,'TPR' : i_tpr} for i_fpr,i_tpr in zip(fpr, tpr)]

dvclive.log_plot('ROC_Curve',
                 datapoints,
                 x = 'FPR',
                 y = 'TPR',
                 template = 'linear',
                 title = 'ROC Curve'
                 )


#dvclive.log_plot("fpr", fpr)
#dvclive.log_plot("tpr", tpr)
#dvclive.log_plot("threshold", roc_thresholds)