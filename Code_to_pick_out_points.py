#!/usr/bin/python
import numpy as np

######################################################################################
##################### Creates a pointer and saves the picked points  #################
######################################################################################

class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """
    def __init__(self):
	self.lastind = 0
	self.selected,  = ax.plot([xs[0]], [ys[0]], 'o', color='yellow', visible=False)

    def onpick(self, event):
	if event.artist!=line: return True
	N = len(event.ind)
	if not N: return True
	# the click locations
	x = event.mouseevent.xdata
	y = event.mouseevent.ydata
	distances = np.hypot(x-xs[event.ind], y-ys[event.ind])
	indmin = distances.argmin()
	dataind = event.ind[indmin]
	f = open('Data_files/a', "a")
	f.write(str(dataind) + '\n')
	f.close()
	self.lastind = dataind
	self.update()

    def update(self):
        if self.lastind is None: return
        dataind = self.lastind
        ax2.cla()
        ax2.plot(X[dataind])
        self.selected.set_visible(True)
        self.selected.set_data(xs[dataind], ys[dataind])
        fig.canvas.draw()

Optimal_values = np.loadtxt('Data_files/Solar_Optimal_values')
Gamma_values = np.loadtxt('Data_files/Solar_Gamma_values')
Std=Optimal_values[::2]
Cf=Optimal_values[1::2]
Gamma=np.array([Gamma_values[30*x:(x+1)*30] for x in range(len(Std))])
Quantiles = np.loadtxt('Data_files/Optimal_quantiles_solar')


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	X = Gamma
	xs = Std
	ys = Cf
	fig, (ax, ax2) = plt.subplots(2, 1)
	ax.set_title('click on point to plot time series')
	line, = ax.plot(xs, ys, 'o', picker=10)  # 5 points tolerance
	browser = PointBrowser()
	fig.canvas.mpl_connect('pick_event', browser.onpick)
	plt.show()
