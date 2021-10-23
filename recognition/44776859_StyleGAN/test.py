import matplotlib as mpl
from matplotlib import gridspec, pyplot as plt, colors, cm

fig = plt.figure(constrained_layout=False, figsize=(15,11))
fig.suptitle(f'Epoch | Batch')
# Parent gridspec.
gs0 = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2,1], width_ratios=[1], figure=fig,
                        left=.02, right=.98, top=.95, bottom=.05)

ax = []
# Top gridspec (images)
gs1 = gs0[0, 0].subgridspec(ncols=4, nrows=2, height_ratios=[1,1], width_ratios=[1,1,1,1],
                            wspace=0, hspace=0)
for i in range(2):
    for j in range(4):
        ax.append(fig.add_subplot(gs1[i, j]))

gs1 = gs0[1, 0].subgridspec(ncols=3, nrows=1, height_ratios=[1], width_ratios=[5,1,1])
ax8 = fig.add_subplot(gs1[0, 0])
ax9 = fig.add_subplot(gs1[0, 1])
ax10 = fig.add_subplot(gs1[0, 2])

ax8.set_title('the following syntax does the same as the GridSpecFromSubplotSpec call above')
ax9.set_title('Weak Generator')
ax9.set_xlabel('Weak Discriminator')
norm = colors.TwoSlopeNorm(vmin=-1, vcenter=-0.1, vmax=1)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='rainbow'),
             cax=ax9, orientation='vertical', label='Adversarial Balance')
ax9.yaxis.set_label_position('left')
ax9.yaxis.tick_left()
ax9.set_yticklabels([])

ax10.set_title('Loss Ratio')



fig.show()