import sys
import pandas as pd
import numpy as np
import re

# python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv
# The script src/main.py should receive these command line options:
# --train for the path to a training corpus
# --test for the path to a test corpus
# --output for the output file

# arguement parsing uses sys library
for i in range(1, len(sys.argv)):
    if sys.argv[i] == "--train":
        train_path = sys.argv[i + 1]
    elif sys.argv[i] == "--test":
        test_path = sys.argv[i + 1]
    elif sys.argv[i] == "--output":
        output_path = sys.argv[i + 1]


#Reading data:
train_data = pd.read_csv(train_path)  #2000 training data points
test_data = pd.read_csv(test_path)    #400 test data points
#print(train_data.head())

#Preprocessing data:
def preprocess_data(data):
    for index, row in data.iterrows():
        tokens = row['tokens'].split()
        head_pos = list(map(int, row['head_pos'].split()))
        tail_pos = list(map(int, row['tail_pos'].split()))

        head_entity = ' '.join([tokens[i] for i in head_pos])
        tail_entity = ' '.join([tokens[i] for i in tail_pos])

        sentence = row['tokens'] + ' ' + head_entity + ' ' + tail_entity
        sentence = sentence.lower()  # uniformly lower case
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)  # remove punctuations

        data.at[index, 'processed'] = sentence # new column 'processed' for processed sentences 

    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)
#print(train_data.head())



#Bag of Words approach:
def BOW(sentences):
    bag = set()  #set of all words in the corpus, no duplicates 
    for sentence in sentences:
        words = sentence.split()
        bag.update(words)    
    bag = list(bag)
    bag_index = {word: i for i, word in enumerate(bag)}    #map each word to a unique index 

    def vector_rep(sentence):   #vectorize sentences 
        vector = np.zeros(len(bag), dtype=int)
        for word in sentence.split():
            if word in bag_index: #word known in the bag
                vector[bag_index[word]] += 1
        return vector

    return vector_rep, bag_index


vectors, index = BOW(train_data['processed']) #BOW on training data

training_vectors = np.array([vectors(sentence) for sentence in train_data['processed']])
testing_vectors = np.array([vectors(sentence) for sentence in test_data['processed']])
#print(training_vectors.shape, testing_vectors.shape)    #about (2000, 1430) (400, 1430)


class NBClassifier:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.bag_size = 0

    def train(self, X, y):
        total_docs = len(X)
        self.bag_size = X.shape[1]

        # Compute class priors
        class_counts = y.value_counts().to_dict()
        self.priors = {c: class_counts.get(c, 0) / total_docs for c in y.unique()}

        # Initialize word likelihoods dictionary
        self.likelihoods = {c: np.ones(self.bag_size) for c in self.priors}

        # Compute word likelihoods
        for c in y.unique():
            class_vectors = X[y == c]
            word_count = class_vectors.sum(axis=0)
            total_word_count = word_count.sum()
            self.likelihoods[c] = (word_count + 1) / (total_word_count + self.bag_size) #using laplace smoothing

    def predict(self, X):
        predictions = []
        for i in X:
            class_probs = {}
            for c in self.priors:
                log_prob = np.log(self.priors[c])
                log_prob += np.sum(np.log(self.likelihoods[c])*i) #log to prevent underflow
                class_probs[c] = log_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions
    
#predictions using the model
nb_model = NBClassifier()
nb_model.train(training_vectors, train_data['relation'])
predictions = nb_model.predict(testing_vectors)

# 3 fold cross validation
# Function to perform 3-fold cross-validation
def k_fold_cross_validation(X, y, k=3):
    fold_size = len(X) // k
    accuracy = []

    for fold in range(k):
        # Split data into training and testing sets for the current fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size

        X_test_fold = X[start_idx:end_idx]
        y_test_fold = y.iloc[start_idx:end_idx]

        X_train_fold = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train_fold = pd.concat([y.iloc[:start_idx], y.iloc[end_idx:]])

        # Train the model on the training fold
        nb_model = NBClassifier()
        nb_model.train(X_train_fold, y_train_fold)

        # Predict on the testing fold
        predictions_fold = nb_model.predict(X_test_fold)

        # Calculate accuracy for the current fold
        correct_predictions = np.sum(y_test_fold == predictions_fold)
        accuracy.append(correct_predictions / len(y_test_fold))

    return accuracy

# Perform 3-fold cross-validation
fold_accuracy_scores = k_fold_cross_validation(training_vectors, train_data['relation'])
# Print accuracy scores for each fold
for i, accuracy_fold in enumerate(fold_accuracy_scores):
    print(f'Fold {i + 1} Accuracy: {accuracy_fold}')

# Print average accuracy across all folds
average_accuracy = np.mean(fold_accuracy_scores)
print(f'Average Accuracy in Validation on Training Data: {average_accuracy}')

print("---------------------------")

correct_predictions = np.sum(test_data['relation'] == predictions)
accuracy = correct_predictions / len(test_data['relation'])
print('Test Data Accuracy:', accuracy)

print("---------------------------")
#Confusion matrix
confusion_matrix = pd.crosstab(test_data['relation'], predictions, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

print("---------------------------")
#Output
output_df = pd.DataFrame({'original_label': test_data['relation'], 'output_label': predictions, 'row_id': test_data['row_id']}) #output format
output_df.to_csv(output_path, index=False)

# Using the table compute the precision and recall both micro and macro averaged
TP = confusion_matrix.values.diagonal().sum()
FP = confusion_matrix.values.sum() - TP
FN = confusion_matrix.values.sum() - TP
micro_precision = TP / (TP + FP)
micro_recall = TP / (TP + FN)
print('Micro-averaged precision:', micro_precision)
print('Micro-averaged recall:', micro_recall)

print("---------------------------")
macro_precision = 0
macro_recall = 0
for i in range(len(confusion_matrix)):
    TP = confusion_matrix.values[i][i]
    FP = confusion_matrix.values[i].sum() - TP
    FN = confusion_matrix.values[:, i].sum() - TP
    macro_precision += TP / (TP + FP)
    macro_recall += TP / (TP + FN)
macro_precision /= len(confusion_matrix)
macro_recall /= len(confusion_matrix)
print('Macro-averaged precision:', macro_precision)
print('Macro-averaged recall:', macro_recall)
print("---------------------------")