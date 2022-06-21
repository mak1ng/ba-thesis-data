import sys
from sklearn import metrics
import matplotlib.pyplot as plt
if len(sys.argv) < 2:
    print("Not enough arguments, pass file name")
    exit(1)

inp = sys.argv[1]
right_predictions = 0
total_predictions = 0
in_top_5 = 0
not_top_most_types_total = 0
not_top_most_types_right = 0
not_top_most_types_in_top5 = 0
any_str = "$typing.Any$"
total_any = 0
right_any = 0
right_any_in_top_5 = 0
top_most_types = ["$builtins.str$", "$builtins.int$", 
    "$typing.List[typing.Any]$", "$None$", "$typing.List[builtins.str]$" ,"$builtins.bool$", "$builtins.float$", "$typing.Dict[typing.Any,typing.Any]$", "$logging.Logger$", "$typing.Dict[builtins.str,typing.Any]$"]
top_most_types_total_number = [0]*10
top_most_types_right_number = [0]*10
top_most_types_in_top_5 = [0]*10
top_most_types_total = 0
top_most_right = 0
top_most_in_top_5 = 0

with open(inp, 'r') as f:
	predictions = [line.rstrip() for line in f]
total_predictions = len(predictions)

def accuracy():
    for p in predictions:
        results = p.split("\t")
        if results[0] == results[2]:
            right_predictions += 1
        if int(results[4]) in range(1,6):
            in_top_5 += 1
        if results[0] in top_most_types:
            top_most_types_total += 1
            top_most_types_total_number[top_most_types.index(results[0])] += 1
            if results[0] == results[2]:
                top_most_types_right_number[top_most_types.index(results[0])] += 1
                top_most_right += 1
            if int(results[4]) in range(1,6):
                top_most_types_in_top_5[top_most_types.index(results[0])] += 1
                top_most_in_top_5 += 1
        elif results[0] != any_str:
            not_top_most_types_total += 1
            if results[0] == results[2]:
                not_top_most_types_right += 1
            if int(results[4]) in range(1,6):
                not_top_most_types_in_top5 += 1
        if results[0] == any_str:
            total_any += 1
            if results[0] == results[2]:
                right_any += 1
            if int(results[4]) in range(1,6):
                right_any_in_top_5 += 1

    print("Right Predictions:", round(right_predictions*100/total_predictions, 2), right_predictions, "von", total_predictions)
    print("In Top 5:", round(in_top_5*100/total_predictions, 2), in_top_5, "von", total_predictions)
    print("Top 10 common Types:", round(top_most_right*100/top_most_types_total, 2), top_most_right, "von",  top_most_types_total)
    print("Top 10 common Types in Top 5 predictions:", round(top_most_in_top_5*100/top_most_types_total, 2), top_most_in_top_5, "von", top_most_types_total)
    print("Not Top 10 common Types:", round(not_top_most_types_right*100/not_top_most_types_total, 2), not_top_most_types_right, "von", not_top_most_types_total)
    print("Not Top 10 most Types in Top 5 preds:", round(not_top_most_types_in_top5*100/not_top_most_types_total, 2), not_top_most_types_in_top5, "von", not_top_most_types_total)
    print("Top most types probability:", top_most_types_right_number, top_most_types_total_number)
    for i in range(len(top_most_types)):
        print(top_most_types[i], round(top_most_types_right_number[i]*100/top_most_types_total_number[i], 2), top_most_types_right_number[i], top_most_types_total_number[i], "top5:", round(top_most_types_in_top_5[i]*100/top_most_types_total_number[i], 2), top_most_types_in_top_5[i], top_most_types_total_number[i])
    print("Any probability:", round(right_any*100/total_any, 2), right_any, total_any)
    print("Any in Top 5:", round(right_any_in_top_5*100/total_any, 2), right_any_in_top_5, total_any)

def precision_top10_types():
    true_positive = 0
    false_positive = 0
    val = "$None$"
    for p in predictions:
        results = p.split("\t")
        if results[0] == val and results[2] == val:
            true_positive += 1
        elif results[0] != val and results[2] == val:
            false_positive += 1
    print("Precision:", true_positive/(true_positive + false_positive))

def accuracy():
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    count = 0
    val = "$typing.Any$"
    for p in predictions:
        results = p.split("\t")
        if results[0] == val:
            count += 1
        if results[0] == val and int(results[4]) in range(1,6):
            true_positive += 1
        elif results[0] != val and int(results[4]) in range(1,6):
            false_positive += 1
        elif results[0] == val and not int(results[4]) in range(1,6):
            false_negative += 1
        elif results[0] != val and not int(results[4]) in range(1,6):
            true_negative += 1
    print(true_positive, true_negative, false_positive, false_negative)

    print("accuracytop5:", (true_positive + true_negative)/(true_positive +true_negative + false_positive + false_negative))

def accuracy_top5():
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    count = 0
    val = "$typing.Any$"
    for p in predictions:
        results = p.split("\t")
        if results[0] == val:
            count += 1
        if results[0] == val and results[2] == val:
            true_positive += 1
        elif results[0] != val and results[2] == val:
            false_positive += 1
        elif results[0] == val and results[2] != val:
            false_negative += 1
        elif results[0] != val and results[2] != val:
            true_negative += 1
    print("accuracy:", (true_positive + true_negative)/(true_positive +true_negative + false_positive + false_negative), count)

def recall_top10_types():
    true_positive = 0
    false_negative = 0
    val = "$builtins.str$"
    for p in predictions:
        results = p.split("\t")
        if results[0] == val and results[2] == val:
            true_positive += 1
        elif results[0] == val and results[2] != val:
            false_negative += 1

    recall = true_positive/(true_positive + false_negative)
    print(recall, true_positive, false_negative)

true_values = []
predicted_values = []
pred_rank = []
for p in predictions:
    results = p.split("\t")
    true_values.append(results[0])
    predicted_values.append(results[2])
    pred_rank.append(results[4])

with open("metrics.txt", "w") as f_out:
    m = metrics.classification_report(true_values, predicted_values, digits=4, output_dict=True)
    for i in range(len(top_most_types)):
        f_out.write(top_most_types[i] + str(m.get(top_most_types[i])) + "\n")

matrix = metrics.confusion_matrix(true_values, predicted_values)
print(matrix)
print(matrix.diagonal()/matrix.sum(axis=1))
cm = metrics.confusion_matrix(true_values[:100], predicted_values[:100], normalize='all')
cmd = metrics.ConfusionMatrixDisplay(cm)
cmd.plot()
plt.savefig('foo.png')