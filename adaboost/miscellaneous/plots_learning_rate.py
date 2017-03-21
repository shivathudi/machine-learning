import matplotlib.pyplot as plt

mean_test_score = [ 0.95139036,  0.9530578 ,  0.95278079,  0.95194669,  0.95166775,
        0.95194977]

mean_train_score = [ 0.99770843,  0.9993056 ,  0.9993056 ,  0.9993056 ,  0.9993056 ,
        0.9993056 ]

learning_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30]


fig, ax = plt.subplots()

x = learning_rates
y = mean_train_score
ax.plot(x, y, 'k', label='Training Accuracy')
fig.suptitle('Training and Cross-Validation Accuracy vs Learning Rate \n (Number of Estimators = 350 and Maximum Depth = 6)', fontsize=12, fontweight='bold')

y = mean_test_score
ax.plot(x, y, 'r', label='Cross-Validation Accuracy')

ax.set_xlabel('Learning Rates')
ax.set_ylabel('Accuracy')

legend = ax.legend(loc='center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width


plt.savefig('gradboost_learning_rates.pdf', format="pdf")