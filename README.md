# 151A Project

## Introduction
We decided to pursue this project for its real-world relevance in addressing the challenge of predicting user ratings from textual reviews. For this project, we decided to focus on the [Amazon Digital_Music](https://amazon-reviews-2023.github.io/) reviews dataset. This topic touches on the intersection of sentiment analysis, focusing on how user-generated content can reveal customer satisfaction. The dataset's inherent challenges, such as class imbalance, make it an interesting and valuable task for machine learning practitioners, while its application in the digital music domain aligns with practical and impactful use cases like personalized recommendations.

We thought this project to be exciting because it shows the power of machine learning techniques to interpret natural language and deliver meaningful insights. A good predictive model in this context can have the following implications: it can enhance customer satisfaction, improve platform loyalty, and support better business strategies by understanding user needs. It also enables user to discover music they love more efficiently. 

## Methods

### Data Exploration
All data exploration can be found here: [Exploration Notebook](https://github.com/EddieJ03/151a-project/blob/Milestone2/DataCleaning.ipynb)

To start exploring the data, we first converted it from a JSON file into a CSV format. After converting, we took a look at the first couple of rows to get a sense of the data:
![First Five Rows](image.png)

We then decided to see what columns and rows had missing values, and how much. This led us to discover the following:
```
rating                0
title                20
text                 23
images                0
asin                  0
parent_asin           0
user_id               0
timestamp             0
helpful_vote          0
verified_purchase     0
```
So it looks like the data is fairly complete, with only some missing data for the title and text columns.

We were then curious about the number of unique values per each column, which led us to find the following:
```
Column 'rating' has 5 unique values.
Column 'title' has 86010 unique values.
Column 'text' has 118641 unique values.
Column 'images' has 3269 unique values.
Column 'asin' has 70519 unique values.
Column 'parent_asin' has 70511 unique values.
Column 'user_id' has 100952 unique values.
Column 'timestamp' has 128745 unique values.
Column 'helpful_vote' has 96 unique values.
Column 'verified_purchase' has 2 unique values.
```

As expected, columns dealing with more variable data such as text and time have the most diversity, whereas the numerical columns have much less unique values.

We noticed the verified_purchase column only had two values, likely for verify and not verified. We were curious how much reviews were verified, and we got the following:
```
True     96033
False    34401
``` 

Finally, we did a pairplot between all the numerical columns to notice any relationships, and got the following:
![pairplot](image-1.png)

Since we are mostly interested in predicing the `rating` column, we decided to take a look at how the other columns might affect it. One thing to notice is that there doesn't seem to be a clear trend between timestamp and rating, suggesting ratings are not strongly correlated with the time of review. Another observation is that a lot of the ratings are 5, if you'll notice the top-left plot. Additionally, the helpful vote values look to be somewhat normally sitributed around the ratings. Another way of looking at it is every rating has at least one helpful vote. This also applies for verified purchase.

### Data Preprocessing
All data preprocessing can be found here: [Preprocessing Notebook](https://github.com/EddieJ03/151a-project/blob/Milestone3/DataCleaning.ipynb)

Our data preprocessing included several steps. First of all, we chose to use only verified reviews, i.e., only reviews from those who actually purrchased the music. Then, we filtered the data columns to only include the title, text, and review columns, getting rid of columns like userid, images, helpful votes, etc.

Next, we had to account for the observations that had a title, but were missing text, and vice versa. In these cases, we just filled in the text with the title if the text was empty, or the title with the text if the title was empty. If both the title and text were missing, we just accounted for this by making some dummy text corresponding to the rating:
```
    df.loc[missing_mask & (df['rating'].isin([4, 5])), ['title', 'text']] = 'very good'
    df.loc[missing_mask & (df['rating'] == 3), ['title', 'text']] = 'good'
    df.loc[missing_mask & (df['rating'].isin([1, 2])), ['title', 'text']] = 'bad'
```

We also cleaned all the text appearing in the 'title' or 'text' columns to remove whitespace, emojis, or html entities that were left over. 

Then, we created columns based off the text and titles, 3 for the title and 3 for the text. The columns we created are 'title_exclamations' (# of exclamation marks), 'title_questions' (# of question marks), 'title_word_count', and the same 3 for the text. 

Next, we made a positive review column, where a rating of 3 or more was considered positive, and a rating lower than 3 was considered negative.

We then trimmed our dataframe to include only the 3 title columns, 3 text columns, and the positive review column, with our X being the title/text columns, and our Y being the positive review column.

Next, we scaled our data, using StandardScaler from sklearn. 

Finally, due to our class imbalance, with many more positive reviews than negative ones, we used a RandomOverSampler, to balance out the classes. 

### Model 1
Our first model can be found in this notebook as well: [Preprocessing Notebook](https://github.com/EddieJ03/151a-project/blob/Milestone3/DataCleaning.ipynb)

For this model, we decided to use a LogisticRegression model, and tried with various regularization values to see if the model improved with more/less complexity. 
```
complexity_values = np.logspace(-3, 3, 20)  # from simple to very complex
train_errors = []
test_errors = []

for C in complexity_values:
    model = LogisticRegression(random_state=42, max_iter=1000, C=C)
    model.fit(X_train_balanced, y_train_balanced)
```
As can be seen in the code snippet above, we decided to tune the model's regularization to see how this affected performance. This not only informed us on the best parameters for logistic regression, but also gave us insight on what to tune for model 2. 


### Model 2
All our second model code can be found here: [All Steps Notebook](https://github.com/EddieJ03/151a-project/blob/Milestone4/DataCleaning.ipynb)

For our second model, we decided to use a TfidfVectorizer to vectorize the text in the reviews, rather than using # of question marks, word count, and # of exclamation marks. For our parameters for this vectorizer, we decided to cap the number of features at 5000, and allowed it to consider single and two-word sequences by setting ngram_range to (1,2). Also note that we combined the text and title together for each review, then vectorized.
```
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
```

We also decided to address our class imbalance using SMOTE rather than RandomOverSampler, as used for Model 1. 
```
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

For the model itself, we chose to use a gradient boosted tree, with the library XGBoost. In terms of the model parameters, we chose to have it use the softmax function to make class probabilities, and made the evaluation metric be log loss. 
```
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(set(y)), 
    eval_metric='mlogloss',
    use_label_encoder=False
)
```
To test various model complexities, we varied the max tree depth and made several different models, with depths from 1 to 12, to ensure we weren't overfitting. 
 
### Model 3
For this model, we chose a new vectorizer, keras.layers.TextVectorization, along with a neural network for classification. 
```
vectorizer = keras.layers.TextVectorization(max_tokens=max_tokens, output_mode='int', output_sequence_length=max_length)
```

For classification, we decided on a 4 hidden layer network, with convolutions and maxpools making up the hidden layers. For the input, we had an embedding layer that converted tokens into a vectorized embedding. For the output, we just had a dense layer that used a sigmoid activation layer to predict class outputs. As an optimizer for training, we chose AdamW. 
```
model = keras.Sequential() 

model.add(keras.layers.Embedding(max_tokens, 32, input_length=max_length)) 

model.add(keras.layers.Conv1D(32, 7, activation='relu'))

model.add(keras.layers.MaxPooling1D(5))

model.add(keras.layers.Conv1D(32, 7, activation='relu'))

model.add(keras.layers.GlobalMaxPooling1D())

model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.AdamW(), metrics=['accuracy']) 
```

We chose to train this model for 5 epochs, with a batch size of 20. 

## Results

### Model 1
We see that for our first model, it seems to do very poorly on our imbalanaced dataset. We see that the training performance achieves a precision of 0.09 for non-positive reviews while achieving a 0.95 precision for positive reviews, with the testing performances doing roughly similar. Overall, we achieved a 67% accuracy on borth training and testing, but a 0.16 F1-score for the negative class (while the positive class has 0.8 F1-score). We see that this logistic regression model does not do well and would need to improve.
## Discussion

### Data Preprocessing
The first key choice we made when doing data preprocessing was to only consider verified reviews for our train/test sets, which we chose as non-verified reviews are not credible and not representative of what someone says when they try a product, as they could've left the review without trying it.

Our next choice was how we decided to fill in missing title/text values. We chose to do this, as stated earliler, with the following scheme:
```
df['title'] = df.apply(lambda x: x['text'] if pd.isna(x['title']) and not pd.isna(x['text']) else x['title'], axis=1)
df['text'] = df.apply(lambda x: x['title'] if pd.isna(x['text']) and not pd.isna(x['title']) else x['text'], axis=1)

# if both title and text are missing fill based on rating
missing_mask = df['title'].isna() & df['text'].isna()
df.loc[missing_mask & (df['rating'].isin([4, 5])), ['title', 'text']] = 'very good'
df.loc[missing_mask & (df['rating'] == 3), ['title', 'text']] = 'good'
df.loc[missing_mask & (df['rating'].isin([1, 2])), ['title', 'text']] = 'bad'
```

The reason we chose to do this was because at this point, for our first model, we wanted to keep separate text and title columns, and needed values for both. Our closest estimation to a user's sentiment when it comes to title/text would be what they put for the other field. However, this choice may have affected our model performance later down the road, as eventually we didn't have separate columns for text and titles, and just vectorized/tokenized the combined text. This means that for the entries that only had title, or only had text, the final combined text just ended up being repeated twice.

In terms of the values we chose for title and text in the case that both were missing, we did this based off of our scale, which had ratings >=3 being a positive review, and scores < 3 being negative. Therefore, it made sense to put sample text corresponding to a negative review for ratings 1 and 2, and sample text corresponding to a positive review for ratings over 3. Also, we knew the choice wasn't very important, as from the Data Exploration, we knew that there were only 40 or so entries with null values for text or title, out of the 130,000 total examples. Thus, we didn't need to waste too much time or resources deciding on the best specific words, and just did something simple that reflected the sentiment for each rating.

Furthermore, we chose to remove HTML entities, emojis, and whitespace from the text to standardize the text, and to remove unnecessary complexity that only appeared in a few reviews, like emojis.

Our main choice for the data was how to represent the text in a way that a Logistic Regression model could reason on. Initially, we chose to see if question mark count, exclamation point count, and word count were enough information, and the data we received from this informed our choices for future models. Since it was our first model, it made sense to start with something simple that could inform our future experiments, and show us how important it is to include information about the specific words rather than just punctuation.

In addition, one of our key issues with our dataset, as seen in Data Exploration, was the class imbalance, with the dataset having way more positive than negative reviews. We decided to use an over sampling method, because this class imbalance had to be addressed for any reasonable model, otherwise a model that only ever predicted positive would have great accuracy. Without oversampling, it would be difficult to train a model that reasons well because it would just trend towards only predicting positive.

Also, given our class imbalance, we wanted to group together ratings of 1 and 2, as there were just not enough observations with a rating of 1 or observations with a rating of 2 to have them be their own classes. This is why we chose to have a positive review column instead of rating. 

For our oversampling method, we started with just RandomOverSampling, which just takes random negative observations and duplicates them to balance out the classes. For our first model, we wanted to start with something simple, and based on our results, then see if more advanced methods were necessary. 

### Model 1
The reason we chose to start with a Logistic Regression model was to 


## Conclusion

## Statement of Collaboration

