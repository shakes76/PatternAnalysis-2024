import csv

def extract_loss(file_path):

	vals = []

	with open(file_path, "r") as log_file:

		contents = log_file.read().split("\n")

	for line in contents:
		loss = line.split("-")[-1]

		try:
			loss = float(loss)
		except:
			continue

		if loss > 3:
			# outlier
			continue

		vals.append(float(loss))

	return vals

if __name__ == "__main__":

	loss = extract_loss("screenlog.0")

	with open("lossplot.csv", "w") as loss_file:

		loss_writer = csv.writer(loss_file)
		for line in enumerate(loss):
			loss_writer.writerow([line[0], line[1]])
