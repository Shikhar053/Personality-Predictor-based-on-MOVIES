#1. Preprocessing of the movies dataset that I found on Kaggle--> The TMDB Dataset of
#5000 movies, It had a lot of object type values and 2 DATASETS acutually so...
import pandas as pd
movies=pd.read_csv("tmdb_5000_movies.csv")
credit=pd.read_csv("tmdb_5000_credits.csv")
movies=pd.merge(movies,credit,left_on ="id",right_on="movie_id")
#print(movies.head())
#movies is our dataset but it needs to be preprocessed  A LOT!!!
#print(movies.info())


#PREPROCESSING
# 1. Keeping the necessary Columns/Features --> Based on MY DOMAIN DOMAIN KNOWLEDGE
cols=["genres","budget","vote_count","crew","original_title","id"]
movies=movies[cols]
#2.Extract Actual Generes from the mess thats provided to me in the name of lists AS STRINGS
import ast
def GETGENRES(genre):
    try:
        genres = ast.literal_eval(genre)
        return [genre['name'] for genre in genres]
    except (ValueError, KeyError, TypeError):
        return []
movies["genres"]=movies["genres"].apply(GETGENRES)
movies = movies[movies["genres"].map(len) > 0]
#print(movies["genres"])
#3. do the same for getting the director and producer of the film from CREW...a bit more tedius
def getCrew(crew,role):
    try:
        crewset=ast.literal_eval(crew)
        for member in crewset:
            if member["job"]==role:
                return member["name"]
        return None
    except  (ValueError, KeyError, TypeError):
        return None
#making the NEW DIRECTOR AND PRODUCER COLUMNS FOR THE NEW PREPROCESSED CSV DATA
movies["director"]=movies["crew"].apply(lambda x: getCrew(x,"Director"))
movies["producer"]=movies["crew"].apply(lambda x: getCrew(x,"Producer"))

##countNone=0
##for i in movies["producer"]:
##    if i==None:
##        countNone+=1
##print("COUNT OF NONE",countNone)

#since 996 values existed where producer was none and 16 where director is none
movies['director'] = movies['director'].replace('None', pd.NA)
movies = movies.dropna(subset=['director'])
movies['producer'] = movies['producer'].replace('None', pd.NA)
movies['producer'] = movies['producer'].fillna('Unknown')

#cleaning budget and vote_count columns to handle missing values efficiently
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce').fillna(0)
movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)
#print(movies)

# Changing Values of Budget to High/Low and vote_count as high/low
#print(movies['budget'].dtype)
med=movies["budget"].median()
def highlowcategory(budget):
    global med
    if budget<med:
        return "Low"
    else:
        return "High"
movies["budget"]=movies["budget"].apply(highlowcategory)
med=movies["vote_count"].median()
def highlowcount(count):
    global med
    if count<med:
        return "Low"
    else:
        return "High"
movies["vote_count"]=movies["vote_count"].apply(highlowcount)
#Making New Personality column: Dependent variable
def personality(genre):
    if "Action" in genre or "Adventure" in genre:
        return "Adventure lover PROBABLY BUNNY FROM YJHD?"
    elif "Drama" in genre or "Romance" in genre:
        return "Emotional CryBaby"
    elif "Comedy" in genre:
        return "Mr/Mrs. Funny Bones"
    else:
        return "Indescribably Weird"
movies["personality"]=movies["genres"].apply(personality)
def personality_label(personality):
    if personality in ["Adventure lover PROBABLY BUNNY FROM YJHD?", "Mr/Mrs. Funny Bones"]:
        return "Positive"
    else:
        return "Negative"
movies["personality"] = movies["personality"].apply(personality_label)
newdataset=movies[['original_title', 'genres', 'budget', 'vote_count', 'director', 'producer', 'personality']]
import os
if os.path.exists("PreprocessedData.csv"):
    print("Data Already Preprocessed!!")
else:
    newdataset.to_csv("PreprocessedData.csv",index=False)
    print("Tmdb5000 Dataset has been Preprocessed Successfully!!!")
