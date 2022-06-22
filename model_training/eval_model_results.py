import sys


if len(sys.argv) < 2:
    print("Not enough arguments, pass file name")
    exit(1)

inp = sys.argv[1]
#top 10 most seen types in training
top_10_most_types = ["$builtins.str$", "$builtins.int$", "$typing.List[typing.Any]$", "$None$", "$typing.List[builtins.str]$" ,
    "$builtins.bool$", "$builtins.float$", "$typing.Dict[typing.Any,typing.Any]$", "$logging.Logger$", 
    "$typing.Dict[builtins.str,typing.Any]$"]

with open(inp, 'r') as f:
	predictions = [line.rstrip() for line in f]
total_predictions = len(predictions)

def top_k_accuracy(k):
    '''Computes and prints the Top-1 and Top-k overall Accuracy'''
    right_predictions = 0
    in_top_k_predictions = 0
    for p in predictions:
        results = p.split("\t")
        if int(results[3]) in range(1, k + 1):
            in_top_k_predictions += 1
        if results[0] == results[1]:
            right_predictions += 1
    print("Total Top-1 Accuracy:", round((right_predictions*100)/total_predictions, 2), "\nTotal Top-" + str(k) + " Accuracy:", round((in_top_k_predictions*100)/total_predictions, 2), in_top_k_predictions, total_predictions)

def top_k_accuracy_top_10_most_types(k):
    '''Computes and prints the Top-1 and Top-k Accuracy for the 10 most seen types in training.'''
    right_predictions_ten = 0
    in_top_k_predictions_ten = 0
    right_predictions_other = 0
    in_top_k_predictions_other = 0
    total_ten = 0
    total_other = 0
    for p in predictions:
        results = p.split("\t")
        if results[0] in top_10_most_types:
            total_ten += 1
            if results[0] == results[1]:
                right_predictions_ten += 1
            if int(results[3]) in range(1, k + 1):
                in_top_k_predictions_ten += 1
        elif results[0] not in top_10_most_types and results[0] != "$typing.Any$":
            total_other += 1
            if results[0] == results[1]:
                right_predictions_other += 1
            if int(results[3]) in range(1, k + 1):
                in_top_k_predictions_other += 1


    print("Top 10 Types Top-1 Accuracy:", round((right_predictions_ten*100)/total_ten, 2), "\nTop 10 Types Top-" + str(k) + " Accuracy:", round((in_top_k_predictions_ten*100)/total_ten, 2))
    print("Other Top-1 Accuracy:", round((right_predictions_other*100)/total_other, 2), "\nOther Top-" + str(k) + " Accuracy:", round((in_top_k_predictions_other*100)/total_other, 2))


def top_1_accuracy(type_label):
    '''Computes and prints the Top-1 accuracy, precision and recall for a specific type_label.'''
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    count = 0
    for p in predictions:
        results = p.split("\t")
        if results[0] == type_label:
            count += 1
        if results[0] == type_label and results[1] == type_label:
            true_positive += 1
        elif results[0] != type_label and results[1] == type_label:
            false_positive += 1
        elif results[0] == type_label and results[1] != type_label:
            false_negative += 1
        elif results[0] != type_label and results[1] != type_label:
            true_negative += 1
    print(type_label, "Top1")
    print("TP:", true_positive, "TN:", true_negative, "FP:", false_positive, "FN:", false_negative)
    print("accuracy: ", round(((true_positive + true_negative)*100)/(true_positive +true_negative + false_positive + false_negative), 2), "Total preds:", count)
    #print("R/T:", round((true_positive*100)/count, 2))
    print("Precision", round((true_positive*100)/(true_positive + false_positive), 2))
    print("Recall", round((true_positive*100)/(true_positive + false_negative), 2))

def top_5_accuracy_specific(type_label):
    '''Computes and prints the Top-5 accuracy, precision and recall for a specific type_label.'''
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    count = 0
    for p in predictions:
        results = p.split("\t")
        top_5_guesses = results[4][1:-1].split('\', \'') #'$typing.Any$'
        top_5_guesses[0] = top_5_guesses[0][1:]
        top_5_guesses[4] = top_5_guesses[4][:-1]
        if results[0] == type_label:
            count += 1
        if results[0] == type_label and type_label in top_5_guesses:
            true_positive += 1
        elif results[0] != type_label and type_label in top_5_guesses:
            false_positive += 1
        elif results[0] == type_label and type_label not in top_5_guesses:
            false_negative += 1
        elif results[0] != type_label and type_label not in top_5_guesses:
            true_negative += 1
    print(type_label, "Top5")
    print("TP:", true_positive, "TN:", true_negative, "FP:", false_positive, "FN:", false_negative)
    print("Accuracy: " + type_label, round(((true_positive + true_negative)*100)/(true_positive +true_negative + false_positive + false_negative), 2), "Total preds:", count)
    #print("R/T:", round((true_positive*100)/count, 2))
    prec = (true_positive)/(true_positive + false_positive)
    recall = (true_positive)/(true_positive + false_negative)
    print("Precision", round((true_positive*100)/(true_positive + false_positive), 2))
    print("Recall", round((true_positive*100)/(true_positive + false_negative), 2))
    print("F1 Score", round(200*((prec * recall)/(prec + recall)), 2))

def precision(type_label):
    true_positive = 0
    false_positive = 0
    count = 0
    for p in predictions:
        results = p.split("\t") 
        if results[0] == type_label and results[1] == type_label:
            true_positive += 1
        elif results[0] != type_label and results[1] == type_label:
            false_positive += 1
    print(true_positive/(true_positive+false_positive))
    print("Precision", type_label, round((true_positive*100)/(true_positive + false_positive), 2))

#Results for the top 10 most seen types in training
for i in range(len(top_10_most_types)):
    top_1_accuracy(top_10_most_types[i])
    top_5_accuracy_specific(top_10_most_types[i])
    print("\n")

#Results for typing.Any
top_1_accuracy("$typing.Any$")
top_5_accuracy_specific("$typing.Any$")
print("\n")
#Results overall
top_k_accuracy(5)
print("\n")
#Results overall for top 10 types
top_k_accuracy_top_10_most_types(5)