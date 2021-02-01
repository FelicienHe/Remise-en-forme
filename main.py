import numpy as np
import csv
from sklearn import svm
import joblib

def main():
    # We build our training set

    # First, we compute the number of line and row in our training set
    row_number = 784
    line_number = 0

    with open('mnist_train.csv', newline='') as f:
        line_number = sum(1 for line in f)

    labels = np.zeros(line_number-1)
    dataset = np.zeros((line_number-1, row_number))

    # Now, we store our training set
    with open('mnist_train.csv', newline='') as csv_file:
        reader = csv.reader(csv_file)
        line_index = 0
        first_line = True

        for line in reader:
            if first_line == False :
                labels[line_index] = line[0]
                dataset[line_index] = line[1:]
                line_index = line_index + 1
            else:
                first_line = False

    print("Maintenant, on va essyer de construire notre model")
    model = svm.SVC(kernel='linear')
    model.fit(dataset, labels)

    print("Le model a bien été construit.")
    # save the model to disk
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

# Now, we execute our code
if __name__ == "__main__":
    main()
