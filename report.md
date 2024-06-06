# CMPUT 461 A5 Report

# Design Decisions and Justifications

The code implements a Naive Bayes text classification model using BOW. It aims to classify relations between entities in text data.
We start off by reading the data from the files and then we preprocess the data by removing stop words, punctuation, and numbers because they don't contribute to the meaning of the sentences, and we mainly care about the head and tail entities. We then create a vocabulary of all the words in the data and then we create a bag of words for unique words for each sentence in the data. 
We structure the code into multiple parts using functions for all these processes as it makes it easier to debug and modify the code and it also makes it easier to understand the code.

We have 3 command line arguments:
1. The first argument is the path to the training data.
2. The second argument is the path to the testing data.
3. The third argument is the path to the output file.

We also have separate python and output files for the normal main.py code and the grad extension - main_grad.py. This is done to make it easier to see the difference in the updated/improved coed and the change in output for the same.  

We use the sys library to read the command line arguments. We then use the arguments to read the data from the files and to write the results to the output file.
We have used the Pandas library to read the data from the files and to store the data in a dataframe due to its efficiency.

## Preprocessing
The preprocessing step is crucial in building the NBClassifier as it allows us to pick out which features we want to focus on. We created a function called preprocess_data that takes in a dataframe and returns a dataframe of preprocessed data. The function performs the following steps:
- It splits the ‘tokens’ into individual words.
- It maps the ‘head_pos’ and ‘tail_pos’ to integers.
- It creates ‘head_entity’ and ‘tail_entity’ by joining the tokens at the positions specified by ‘head_pos’ and ‘tail_pos’.
- It creates a sentence by joining the ‘tokens’, ‘head_entity’, and ‘tail_entity’.
- It converts the sentence to lower case to ensure uniformity.
- It removes punctuations from the sentence using a regular expression.
- Finally, it adds the processed sentence to a new column ‘processed’ in the data.

This preprocessing step is important as it cleans the data and transforms it into a format that can be used for further creating the Naive Bayes model. We have used regular expressions for most cleaning tasks as they are very efficient and easy to use and we are well versed with them.

## Bag of Words Approach

The idea is to create a "bag" (set) of all unique words in the text corpus, and then represent each sentence or document as a vector in this high-dimensional space. Each dimension corresponds to a unique word in the bag, and the value at that dimension in the vector represents the count of that word in the sentence/document.

Reasons for implementing it this way:

- The use of a set to store the bag of words ensures that each word is stored only once, reducing memory usage and increasing efficiency. It was also more efficient to convert the set to a list and the creation of a dictionary allows for efficient lookup using word indices.  

- The function returns a vector_rep function that can be used to vectorize any sentence, this makes it flexible enough to vectorize new data in the same space.  


## Naive Bayes Classifier 
- **train(self, X, y):**
  - The `train` method is responsible for training the Naive Bayes Classifier using the input training data `X` and labels `y`.
  - It computes class priors, initializes the word likelihoods dictionary, and calculates word likelihoods for each class.
  - Laplace smoothing is applied to handle the problem of zero probabilities.

- **predict(self, X):**
  - The `predict` method takes a set of input data `X` and returns a list of predicted class labels.
  - For each input instance, it calculates the log probabilities for each class and predicts the class with the maximum log probability.
  - Logarithms are used to prevent underflow issues.

Example Usage:

```python
# Instantiate the Naive Bayes Classifier
nb_classifier = NBClassifier()

# Train the classifier
nb_classifier.train(training_data, training_labels)

# Make predictions on test data
predictions = nb_classifier.predict(test_data)
```

This implementation allows us to easily train and test the Naive Bayes Classifier on different datasets. It also allows us to easily modify the classifier to implement different techniques, such as add-k smoothing, lidstone smoothing, etc. We can also easily modify the classifier to implement different feature selection techniques, such as skip most common words, etc.

# Performance Report

## Accuracy
| Fold | Training Data Accuracy |
|------|------------------------|
| 1    | 0.8844                 |
| 2    | 0.8694                 |
| 3    | 0.8784                 |
| Cross-Validation Avg  | 0.8774                 |
| Test Data Accuracy | 0.885                  |

We were able to achieve an accuracy of 88.7 after training the model on the training data and cross validating it using a 3-fold CV. We created a function called k_fold_cross_validation that takes in the training data and splits it into k equally sized folds, then use k-1 folds for training and the remaining fold for testing. This process is repeated k times, each time using a different fold for testing. The performance of the model is then averaged over the k iterations to give a more robust estimate of its performance.
- we are using numpy and pandas operations for efficient data manipulation.  
- we've also made the function data-agnostic. Since X (feature matrix) and y (target vector) are inputs to the function, it can work with any held-out dataset as well. The user can also update the number of folds (k) independently.  



After cross validation on our training data we were able to achieve an average accuracy of 88.5 on the testing data. 

## Confusion Matrix
|            | Predicted Characters | Predicted Director | Predicted Performer | Predicted Publisher |
|------------|----------------------|--------------------|---------------------|---------------------|
| Actual Characters   | 92                   | 5                  | 4                   | 2                   |
| Actual Director     | 7                    | 76                 | 8                   | 3                   |
| Actual Performer    | 5                    | 5                  | 90                  | 3                   |
| Actual Publisher    | 3                    | 1                  | 0                   | 96                  |

The confusion matrix shows the number of times the model predicted a certain relation when the actual relation was something else. We have used the Pandas library to create the confusion matrix as it's crosstab method makes a the matrix and is easy to use. 

## Precision and Recall
| Metric                      | Value                |
|-----------------------------|----------------------|
| Micro-averaged Precision    | 0.885                |
| Micro-averaged Recall       | 0.885                |
| Macro-averaged Precision    | 0.8838752323900021   |
| Macro-averaged Recall       | 0.8847015416890869   |


# Error Analysis 

### Sources of Error

1. **Ambiguity in Entity Relations:**
   - The confusion between different classes in your matrix could be due to this ambiguity. For instance, the model misclassified 5 actual ‘Characters’ as ‘Director’ and 4 as ‘Performer’. This could be because the same person could be a character, a performer, or a director, leading to confusion.

2. **Lack of Contextual Information:**
   - The Bag of Words (BOW) model does not consider the contextual information between words in a sentence. This could explain why the model misclassified 7 actual ‘Directors’ as ‘Characters’ and 8 as ‘Performers’. A more context-aware model might be able to better distinguish these classes.

3. **Overfitting on Training Data:**
   - The model could be overfitting the training data, leading to a lower accuracy on the test data.

### Possible Improvements

1. **Fine-tune Preprocessing:**
   - Review and fine-tune the preprocessing steps to ensure that essential information is not lost, and the model receives the most relevant features.

2. **Address Overfitting:**
   - Implement regularization techniques, such as dropout or adjusting hyperparameters, to prevent overfitting and improve the model's ability to generalize.

3. **Skip Most Common Words:**
   - Skip the most common words in the corpus to focus on more informative and distinctive terms.

4. **Implement Add-k Smoothing:**
    - Implement add-k smoothing to modify the model to handle the problem of zero probability of a word in a class.


# Graduate Student Extension

## Methods of Improving the Model

### 1. Skipping the most common words
This method can help address the issue of lack of contextual information. As we are using the bag of words approach, we are not considering the context of the words in the sentence and some words may exist in most of the sentences but not contribute to the effectiveness of the model.
In order to enhance the effectiveness of the BOW approach, we introduced the ability to skip the most common words. The rationale behind this is to eliminate frequently occurring words that may not contribute much to the understanding of the content. The skip_top parameter allows for the exclusion of the top N most common terms from the bag of words. This helps in focusing on more informative and distinctive terms within the corpus.

### 2. Implementing add-k smoothing
We modified our model to implement add-k smoothing. This is a technique that is used to smooth categorical data. It is used to handle the problem of zero probability. In our case, it is used to handle the problem of zero probability of a word in a class. This is done by adding a constant k to the numerator and adding kV to the denominator, where V is the number of words in the vocabulary. This ensures that the probability of a word in a class is never zero. We have used a value of k = 3.

## Accuracy of the Improved Model

## Validation Performance
| Fold                      | Accuracy               |
|---------------------------|------------------------|
| Fold 1                    | 0.8859                 |
| Fold 2                    | 0.8724                 |
| Fold 3                    | 0.8814                 |
| **Average Accuracy**      | **0.8799**             |
| **Test Data Accuracy** | **0.895**                 |

---

## Confusion Matrix
| Actual\Predicted | characters | director | performer | publisher |
|------------------|------------|----------|------------|-----------|
| characters       | 95         | 2        | 3          | 3         |
| director         | 8          | 77       | 6          | 3         |
| performer        | 5          | 5        | 91         | 2         |
| publisher        | 4          | 1        | 0          | 95        |

---

## Micro-averaged Metrics
**Micro-averaged Precision:** 0.895

**Micro-averaged Recall:** 0.895

---

## Macro-averaged Metrics
**Macro-averaged Precision:** 0.8937

**Macro-averaged Recall:** 0.8966


# Interpretation of Results

We found that implementing the above two methods improved the performance of our model. Further tinkering with the parameters will lead to better results. Although this increase is not very significant and further implementing better feature selection would lead to even further increase in the accuracy of the model. 
