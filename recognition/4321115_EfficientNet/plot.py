from matplotlib  import pyplot as plot
import csv
import sys

def plot_loss(path):
	header, data = [], []
	with open('./'+path, 'r') as file:
		reader = csv.reader(file, delimiter=',')
		count = 0
		for row in reader:
			header.append(row) if count == 0 else data.append(row)
			count += 1
		header = header[0]
	plot.plot([x[1] for x in data], label=header[1])
	plot.plot([x[2] for x in data], label=header[2])
	plot.plot([x[3] for x in data], label=header[3])
	plot.plot([x[4] for x in data], label=header[4])
	plot.legend()
	plot.savefig('./loss.jpg')
	plot.close()

plot_loss(sys.argv[1])