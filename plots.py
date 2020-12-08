import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_json("output_100000/metrics.json", lines=True)

# df.columns:
# ['data_time', 'eta_seconds', 'fast_rcnn/cls_accuracy',
#        'fast_rcnn/false_negative', 'fast_rcnn/fg_cls_accuracy', 'iteration',
#        'loss_box_reg', 'loss_cls', 'loss_rpn_cls', 'loss_rpn_loc', 'lr',
#        'roi_head/num_bg_samples', 'roi_head/num_fg_samples',
#        'rpn/num_neg_anchors', 'rpn/num_pos_anchors', 'time', 'total_loss']

everyN = 50 # plot every 20*everyN interations
df_everyN = df.iloc[::everyN, :]
# print(df_everyN.head(10))
ax1 = df_everyN.plot(kind='line', x='iteration', y='fast_rcnn/cls_accuracy', 
        title='Training Accuracy and Loss')
df_everyN.plot(ax=ax1, kind='line', x='iteration', y='total_loss')
ax1.set_xlabel("Iteration")
plt.show()

ax2 = df_everyN.plot(kind='line', x='iteration', y='total_loss',
        title='All Training Losses')
df_everyN.plot(ax=ax2, kind='line', x='iteration', y='loss_cls')
df_everyN.plot(ax=ax2, kind='line', x='iteration', y='loss_rpn_cls')
df_everyN.plot(ax=ax2, kind='line', x='iteration', y='loss_rpn_loc')
ax2.set_xlabel("Iteration")
plt.show()