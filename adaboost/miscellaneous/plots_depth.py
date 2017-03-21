import matplotlib.pyplot as plt

mean_test_score = [ 0.94333942,  0.95000417,  0.94666658,  0.95055664,  0.95111258,
        0.95166659,  0.95278195,  0.95139228,  0.94944707]

mean_train_score = [ 0.96423632,  0.99354174,  0.99902785,  0.9993056 ,  0.9993056 ,
        0.9993056 ,  0.9993056 ,  0.9993056 ,  0.9993056 ]

depths = range(1,10)


fig, ax = plt.subplots()

x = depths
y = mean_train_score
ax.plot(x, y, 'k', label='Training Accuracy')
fig.suptitle('Training and Cross-Validation Accuracy vs Maximum Depth of Trees \n (Learning Rate = 0.25 and Number of Estimators = 350)', fontsize=12, fontweight='bold')

y = mean_test_score
ax.plot(x, y, 'r', label='Cross-Validation Accuracy')

ax.set_xlabel('Maximum Depth of Trees')
ax.set_ylabel('Accuracy')

legend = ax.legend(loc='center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')

for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width


plt.savefig('gradboost_depths.pdf', format="pdf")