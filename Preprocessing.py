import pandas as pd
import dateutil.parser
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
import ast
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer



# Main preprocessing function
def preprocess(df: pd.DataFrame, is_train: bool = True):
    df = convert_columns_types(df)
    df = extract_top_n_items_from_lists(df, 5, is_train=is_train)[0]
    df = preprocess_numerical(df)
    df = preprocess_categorical(df)
    df = preprocess_dates(df)
    df = preprocess_text(df)
    df = encode_columns(df)
    df = scale_columns(df, is_train= is_train)
    df = drop_columns(df)
    df = impute_columns(df, is_train=is_train)
    return df


def preprocess_numerical(df: pd.DataFrame):
    # Extracting boolean features:
    df['hasBudget'] = [0 if x == 0 else 1 for x in df['budget']]
    df['hasRevenue'] = [0 if x == 0 else 1 for x in df['revenue']]

    # Extracting range related category features
    df['voteCountCategory'] = pd.cut(df['vote_count'], bins=[0, 100, 500, 3000, 7000, 20000], labels=[1, 2, 3, 4, 5])
    df['runtimeCategory'] = pd.cut(df['runtime'], bins=[0, 100, 120, 500], labels=[1, 2, 3])


    return df


def preprocess_categorical(df: pd.DataFrame):
    # Counting list objects
    df['spoken_languages_count'] = df['spoken_languages'].apply(len)
    df['genres_count'] = df['genres'].apply(len)
    df['keywords_count'] = df['keywords'].apply(len)
    df['production_companies_count'] = df['production_companies'].apply(len)
    df['production_countries_count'] = df['production_countries'].apply(len)

    return df


def preprocess_dates(df: pd.DataFrame):
    df['release_date'] = [dateutil.parser.parse(date) if date is not np.nan else pd.NaT for date in df['release_date']]
    df['release_date'] = [date.timestamp() for date in df['release_date']]
    return df


def preprocess_text(df: pd.DataFrame, is_train: bool = True):
    columns = ['original_title', 'overview', 'tagline']
    for column in columns:
        # Apply NLP preprocessing pipeline
        df[column] = df[column].apply(lambda x: NLP_pipeline(x))
        
        # Apply TF-IDF vectorization and convert the result to a dense array
        tfidf_matrix = TF_IDF_vectorize(df[column], column, is_train)
        dense_matrix = tfidf_matrix.toarray()
        
        # Create a new DataFrame from the dense matrix with appropriate column names
        tfidf_df = pd.DataFrame(dense_matrix, columns=[f"{column}_tfidf_{i}" for i in range(dense_matrix.shape[1])])
        
        # Reset the index of the original DataFrame to avoid index mismatches
        df.reset_index(drop=True, inplace=True)
        
        # Concatenate the new DataFrame to the original DataFrame
        df = pd.concat([df, tfidf_df], axis=1)
        
        # Drop the original column to avoid duplication
        df.drop(column, axis=1, inplace=True)
    return df

# Utils:

# Scale numerical data for faster and more efficient convergence
def scale_columns(df: pd.DataFrame, is_train: bool = False, scaler = StandardScaler()):
    columns = ['spoken_languages_count', 'keywords_count', 'genres_count', 'production_companies_count', 'production_countries_count', 'viewercount', 'revenue', 'runtime', 'vote_count', 'budget', 'release_date', 'original_language']
    df[columns] = df[columns].astype(float)
    if is_train:
        df[columns] = scaler.fit_transform(df[columns])
        # Save using pickle
        with open('Pickled/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        scaler = pickle.load(open('Pickled/scaler.pkl', 'rb'))
        df[columns] = scaler.transform(df[columns])
    return df

# Encodes categorical columns to get a numerical format.
def encode_columns(df: pd.DataFrame, is_train: bool = False, encoder = LabelEncoder()):
    columns = 'original_language'    
    df[columns] = df[columns].astype(str)
    
    if is_train:
        df[columns] = encoder.fit_transform(df[columns])
        # Save using pickle
        with open('Pickled/encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
    else:
        encoder = pickle.load(open('Pickled/encoder.pkl', 'rb'))
        df[columns] = encoder.transform(df[columns])
    return df

# Impute function to fill missing values
def impute_columns(df: pd.DataFrame, is_train: bool = True, imputer = SimpleImputer(strategy='most_frequent')):
    # Select only numeric columns for imputation    
    if is_train:
        df_imputed= imputer.fit_transform(df)
        # Save the imputer model using pickle
        with open('Pickled/mean_imputer.pkl', 'wb') as f:
            pickle.dump(imputer, f)
    else:
        # Load the imputer model using pickle
        with open('Pickled/mean_imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        df_imputed = imputer.transform(df)
    
    # Convert the array back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

    return df_imputed

# Takes a string that has a list of object and converts it to just that, an actual list of objects.
def convert_to_list_of_objects(string_representation):
    try:
        return ast.literal_eval(string_representation)
    except (SyntaxError, ValueError):
        return []
    
# Converts all list of objects columns from string to actual lists.
def convert_columns_types(df: pd.DataFrame):
    # Converting string object columns to list representation:
    df['spoken_languages'] = df['spoken_languages'].apply(convert_to_list_of_objects)
    df['genres'] = df['genres'].apply(convert_to_list_of_objects)
    df['keywords'] = df['keywords'].apply(convert_to_list_of_objects)
    df['production_companies'] = df['production_companies'].apply(convert_to_list_of_objects)
    df['production_countries'] = df['production_countries'].apply(convert_to_list_of_objects)
    
    # Extracting the 'name' property from each object in the lists
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['production_companies'] = df['production_companies'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['production_countries'] = df['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    return df

# Extracts top N common features from lists and makes a new column for each feature that is binary encoded.
def extract_top_n_items_from_lists(df, top_n, is_train= True):
    columns = ['spoken_languages', 'genres', 'keywords', 'production_companies', 'production_countries']
    top_items_dict = {}

    # Make sure the directory exists
   

    for column_name in columns:
        pickle_file = f'Pickled/{column_name}_top_{top_n}_items.pkl'

        if is_train:
            # Flatten the list of all items in the column and calculate the frequency of each item
            all_items = [item for sublist in df[column_name].dropna() for item in sublist]
            item_counts = pd.Series(all_items).value_counts().head(top_n)

            # Get the top N items
            top_items = item_counts.index.tolist()
            top_items_dict[column_name] = top_items

            # Save the top items using pickle
            with open(pickle_file, 'wb') as f:
                pickle.dump(top_items, f)

        else:
            # Load the top items from the pickle file
            with open(pickle_file, 'rb') as f:
                top_items = pickle.load(f)
            top_items_dict[column_name] = top_items

        # Create new columns for each top item, indicating presence (1) or absence (0) of the item
        for item in top_items:
            df[item] = df[column_name].apply(lambda x: 1 if item in x else 0)

    return df, top_items_dict

# Applies simple NLP preprocessing pipeline
def NLP_pipeline(text):
    # Make sure text is a string
    if not isinstance(text, str):
        return ''
    # Define tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text and transform to lower cass
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Lemmatize/stem
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Rejoin tokens into a string
    return ' '.join(lemmatized_tokens)

# Feature extraction using TF-IDF
def TF_IDF_vectorize(x, column_name, is_train = True):
    if (is_train):
        vectorizer = TfidfVectorizer(max_features= 10)
        x = vectorizer.fit_transform(x)
        # Save using pickle
        with open(f'Pickled/vectorizer_{column_name}.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        vectorizer = pickle.load(open(f'Pickled/vectorizer_{column_name}.pkl', 'rb'))
        x = vectorizer.transform(x)
    return x


# Drops unwatned columns
def drop_columns(df: pd.DataFrame):
    # Unnecessary columns:
    df.drop('homepage', axis=1, inplace=True) # Irrelevant
    df.drop('status', axis=1, inplace=True) # Has one value so it doesn't differentiate and has no value.
    df.drop('id', axis=1, inplace=True) # Irrelevant
    
    df.drop('title', axis=1, inplace=True)  # Same as original title column
    
    # Features already extracted and columns no longer needed:
    df.drop('spoken_languages', axis=1, inplace=True) 
    df.drop('genres', axis=1, inplace=True) 
    df.drop('keywords', axis=1, inplace=True) 
    df.drop('production_companies', axis=1, inplace=True) 
    df.drop('production_countries', axis=1, inplace=True) 
    
    return df



if __name__ == '__main__':
    df = pd.read_csv('movies-regression-dataset.csv')
    df = preprocess(df)
    df.to_csv('preprocessed_all.csv', index=False, encoding='utf-8-sig')
    print('Data preprocessed')
