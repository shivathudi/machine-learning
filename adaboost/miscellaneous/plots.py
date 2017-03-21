import matplotlib.pyplot as plt

mean_test_score = [0.94722639, 0.9513946, 0.95083557, 0.95223063, 0.95195286,
                          0.95167238, 0.95111644, 0.95306165, 0.95000455]

mean_train_score = [0.9983335, 0.9993056, 0.9993056, 0.9993056, 0.9993056,
                           0.9993056, 0.9993056, 0.9993056, 0.9993056]

tree_list = [50, 100, 150, 200, 250, 300, 350, 400, 450]


fig, ax = plt.subplots()

x = tree_list
y = mean_train_score
ax.plot(x, y, 'k', label='Training Accuracy')
fig.suptitle('Training and Cross-Validation Accuracy vs Number of Trees \n (Learning Rate = 0.25 and Maximum Depth = 6)', fontsize=12, fontweight='bold')

y = mean_test_score
ax.plot(x, y, 'r', label='5-fold Cross-Validation Accuracy')

ax.set_xlabel('Number of Trees')
ax.set_ylabel('Accuracy')

legend = ax.legend(loc='center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width


plt.savefig('gradboost_trees.pdf', format="pdf")