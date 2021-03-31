import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize


# ----------------------------------------------------------------------------
# Function to clean reviews
# ----------------------------------------------------------------------------


def clean_reviews(reviews, stopwords):
    
    '''
    Cleans reviews by making them lowercase, removing digits, special 
    characters and stopwords.
    
    Parameters
    ----------
    reviews : Series (str)
        The text to be cleaned
    stopwords : set, list (str)
        The list of stopwords to remove from the text
        
    Returns
    -------
    reviews : str
        Cleaned review
    '''
    
    # Make lowercase
    reviews = reviews.lower()
    
    # Remove digits and special characters
    reviews = re.sub('[^a-z]',' ', reviews)
    
    # Remove stop words
    reviews = reviews.split()
    reviews = " ".join([word for word in reviews if word not in stopwords])
    
    return reviews


# ----------------------------------------------------------------------------
# Function to create wordcloud
# ----------------------------------------------------------------------------


def create_wordcloud(reviews, stopwords):
    
    '''
    Creates a wordcloud from the words contained within all reviews.
    
    Parameters
    ----------
    reviews : Series (str)
        The text to be used to generate the wordcloud
    stopwords : set, list (str)
        The list of stopwords to be removed from the text before generating
        the wordcloud
    '''
    
    # Set up text
    text = " ".join(reviews.values)
    wordcloud = WordCloud(random_state=2, background_color="white", width=600, 
                          height=350, stopwords=stopwords)
    wordcloud.generate(text)
    
    # Display generated image
    plt.figure(figsize=[10,10])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off");
    
    
# ----------------------------------------------------------------------------
# Function to plot length of reviews by word count
# ----------------------------------------------------------------------------
    

def plot_word_count(reviews, bins, max_x, xtick_dist):
    
    '''
    Plots the length of each review by word count.
    
    Parameters
    ----------
    reviews : Series (str)
        The text to be used for plotting
    bins : int
        Number of bins to use for plot
    max_x : int
        Maximum value for x-axis
    xtick_dist : int
        Distance between x-ticks
    '''
    
    # Get length of each review
    review_length = [len(x.split()) for x in reviews]
    
    # Plot word count including lines for min, median, mean and max
    plt.figure(figsize=(14, 4))
    ax = sns.histplot(review_length, bins=bins, zorder=2)
    ax.set_xticks(range(0, max_x, xtick_dist))
    ax.axvline(np.mean(review_length), color='red', label='Mean')
    ax.axvline(np.median(review_length), color='orange', label='Median')
    ax.axvline(np.max(review_length), color='green', label='Max')
    ax.axvline(np.min(review_length), color='blue', label='Min')
    plt.grid(alpha=.5)
    plt.legend(loc='upper center')
    plt.title('Frequency of Length of Reviews by Word Count', fontweight='bold')
    plt.xlabel('Number of Words');
    

# ----------------------------------------------------------------------------
# Function to plot most common words
# ----------------------------------------------------------------------------


def top_words(reviews, n=25):
    
    '''
    Plots n most common words contained within all reviews.
    
    Parameters
    ----------
    reviews : Series (str)
        The text to be cleaned
    n : int
        The number of words to display (default = 25)
    '''
    
    # Join all words from reviews, tokenize and count them
    all_reviews_joined = " ".join(reviews.values)
    all_tokens = word_tokenize(all_reviews_joined)
    top = Counter(all_tokens).most_common(n)
    
    # Put counts into a dataframe and plot
    pd.DataFrame(top).set_index([0]).plot(kind='bar', rot=45, figsize=(14,4), 
                                          zorder=2, legend=False)
    plt.title(f'Top {n} Words by Frequency', fontweight='bold')
    plt.ylabel('frequency')
    plt.xlabel('word')
    plt.grid(alpha=.25);
    

# ----------------------------------------------------------------------------
# Function for plotting the accuracy and loss for training and validation data
# ----------------------------------------------------------------------------


def acc_loss_plot(history, epochs):
    
    '''
    Plots the accuracy and loss for the training and validation data.
    
    Parameters
    ----------
    history : obj
        The history object which contains a dict of accuracy and loss values
        for each epoch
    epochs : int
        The range of the x axis as determined by the number of epochs
    '''
    
    # Set up figure
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    
    # Plot training and validation accuracy
    axs[0].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Training and Validation Accuracy')
    axs[0].set_xticks(range(0, epochs, 2))
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')

    # Plot training and validation loss
    axs[1].plot(history.history['loss'], label='Training Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Training and Validation Loss')
    axs[1].set_xticks(range(0, epochs, 2))
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss');