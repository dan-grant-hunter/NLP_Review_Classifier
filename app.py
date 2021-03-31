import pickle
from helper_functions import clean_reviews
from model_trainer import TrainModel
from nltk.stem import WordNetLemmatizer
import streamlit as st

# Load star image for reviews
from PIL import Image
image = Image.open('images/star.png')

# ----------------------------------------------------------------------------
# Loading model and function for preparing review
# ----------------------------------------------------------------------------


# Load stopwords
stop_words = pickle.load(open('data/final_stop_words.pkl', 'rb'))


# Create lemmatizer
lem = WordNetLemmatizer()


# Load model
@st.cache(allow_output_mutation=True, show_spinner=False)
def load_model():
    model = pickle.load(open('models/StackingClassifier.pkl', 'rb'))
    return model.best_estimator

# Prepare review for model
def prepare_review(review, stop_words):
    
    # Clean and tokenize review
    review = clean_reviews(review, stopwords=stop_words).split()
    
    # Lemmatize review
    review = ' '.join([lem.lemmatize(word) for word in review])
    
    return [review]


# ----------------------------------------------------------------------------
# Streamlit
# ----------------------------------------------------------------------------

st.title("Hotel Review Classifier")

review = st.text_area('Enter Review Here', height=250)

if st.button('Predict Rating'):
    
    # Prepare review
    review = prepare_review(review, stop_words)
    
    # Make predicition
    model = load_model()
    
    with st.spinner('Reading review...'):
        pred = model.predict(review)[0]
    
    # Return predicted rating
    if pred <=2:
        st.error(pred)
    elif pred == 3:
        st.warning(pred)
    else:
        st.success(pred)

    # Show star rating
    st.image([image]*pred)
