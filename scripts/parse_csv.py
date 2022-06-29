import csv
import pdb
import numpy as np
import pandas as pd

if __name__ == "__main__":
  with open('../files/lumbar-test-pred.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    columns = None 
    counter = 0
    rows = []
    for row in spamreader:
      if counter == 0:
        columns = row 
      else:
        for j in range(len(row)):
          el = row[j]
          if 'score' in columns[j]:
            # This is an array.
            if el == '':
                row[j] = np.array([-1,-1,-1,-1])[:,None]
            else:
              el = el.replace('[',"").replace(']',"").split(",")
              el = np.array([float(scoreString) for scoreString in el])[:,None]
              el = el/el.sum()
              row[j] = el
          else:
            if el == '':
              row[j] = -1
            else:
              row[j] = float(el)
        rows = rows + [row,]
      counter += 1
    df = pd.DataFrame(rows,columns=columns)

    scores = []
    labels = []
    for column in columns:
      if "true" in column:
        labels = labels + [df[column].to_numpy()[:,None],]
      elif "score" in column: 
        scores = scores + [np.concatenate(df[column].tolist(),axis=1).T[:,None,:],]
    scores = np.concatenate(scores,axis=1)
    labels = np.concatenate(labels,axis=1)
    np.save('../files/scores.npy',scores)
    np.save('../files/labels.npy',labels)
