from adaboost import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

train_file = 'hw2_data/spambase.train'
test_file = 'hw2_data/spambase.test'

X, Y = parse_spambase_data(train_file)
Y2 = np.array(new_label(Y))

max_trees = 2000

training_accuracy = []
cross_validation_accuracy = []

tree_list = [1] + range(10, max_trees, 10)

for i in tree_list:
    num_trees = i
    print "Considering %s trees" % (num_trees)
    trees, weights = adaboost(X, Y2, num_trees)
    Yhat = old_label(adaboost_predict(X, trees, weights))

    train_accuracy = accuracy(Y, Yhat)

    training_accuracy.append(train_accuracy)

    # print "Training Accuracy is %s" % (train_accuracy)

    kf = KFold(n_splits=5)
    cv_accuracy_list = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y2[train_index], Y2[test_index]

        trees, weights = adaboost(X_train, y_train, num_trees)
        Yhat = old_label(adaboost_predict(X_test, trees, weights))

        now_accuracy = accuracy(np.array(old_label(y_test)), np.array(Yhat))
        cv_accuracy_list.append(now_accuracy)

    cv_acc = np.mean(cv_accuracy_list)

    # print "10-Fold Cross_validation Accuracy is %s" % (cv_acc)

    cross_validation_accuracy.append(cv_acc)


print tree_list
print training_accuracy
print cross_validation_accuracy


fig, ax = plt.subplots()

x = tree_list
y = training_accuracy
ax.plot(x, y, 'k', label='Training Accuracy')
fig.suptitle('Training and Cross-Validation Accuracy vs Number of Trees', fontsize=14, fontweight='bold')

# plt.title("Training and Cross-Validation Error vs Number of Trees", fontsize=16)
y = cross_validation_accuracy
ax.plot(x, y, 'r', label='Cross-Validation Accuracy')

ax.set_xlabel('Number of Trees')
ax.set_ylabel('Accuracy')

legend = ax.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width


plt.savefig('part2.pdf', format="pdf")


