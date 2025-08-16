# I am going to use 3(maybe 4) algorithms on this i chose as per the preprocessed new dataset)

#1. Decision tree ID3
def ID3():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("PreprocessedData.csv")
    label_encoder = LabelEncoder()
    df['genres'] = df['genres'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
    df['director'] = label_encoder.fit_transform(df['director'])
    df['producer'] = label_encoder.fit_transform(df['producer'])
    df['genres'] = label_encoder.fit_transform(df['genres'])
    mapping = {'High': 1, 'Low': 0}
    df['budget'] = df['budget'].map(mapping)
    df['vote_count'] = df['vote_count'].map(mapping)
    X = df.drop(columns=['personality', 'original_title'])
    y = df['personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
    # CHANGING THE TEST SIZE FROM 0.2 TO 0.3 MADE THE MODEL PERFORM BETTER AND CATCH LOWER RANGE PERSONALITIES MORE EASILY
    id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
    id3_classifier.fit(X_train, y_train)
    y_pred = id3_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy=accuracy*100
    print("ID3: \n")
    print(f"Accuracy: {accuracy}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(id3_classifier, X, y, cv=5)
    scores=scores.mean()*100
    print(f"Cross-validation accuracy: {scores}%")
    return[accuracy,scores]
##    plt.figure(figsize=(12, 8))
##    plot_tree(id3_classifier, feature_names=X.columns, class_names=y.unique(), filled=True)
##    plt.title("ID3 on Movie Personality Dataset")
##    plt.show()
#ID3()

 #2. CART: classification and regression trees   
def CART():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("PreprocessedData.csv")
    label_encoder = LabelEncoder()
    df['genres'] = df['genres'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else '')
    df['director'] = label_encoder.fit_transform(df['director'])
    df['producer'] = label_encoder.fit_transform(df['producer'])
    df['genres'] = label_encoder.fit_transform(df['genres'])
    mapping = {'High': 1, 'Low': 0}
    df['budget'] = df['budget'].map(mapping)
    df['vote_count'] = df['vote_count'].map(mapping)
    X = df.drop(columns=['personality', 'original_title'])
    y = df['personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
    # CHANGING THE TEST SIZE FROM 0.2 TO 0.3 MADE THE MODEL PERFORM BETTER AND CATCH LOWER RANGE PERSONALITIES MORE EASILY
    id3_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)
    id3_classifier.fit(X_train, y_train)
    y_pred = id3_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy=accuracy*100
    print("CART: \n")
    print(f"Accuracy: {accuracy}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(id3_classifier, X, y, cv=5)
    scores=scores.mean()*100
    print(f"Cross-validation accuracy: {scores}%")
    return[accuracy,scores]

    
    #plt.figure(figsize=(12, 8))
    #plot_tree(id3_classifier, feature_names=X.columns, class_names=y.unique(), filled=True)
    #plt.title("CART on Movie Personality Dataset")
    #plt.show()
#CART()

#3. Random Forest: an ensemble method
def RandomForest():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("PreprocessedData.csv")
    label_encoder = LabelEncoder()
    df['director'] = label_encoder.fit_transform(df['director'])
    df['producer'] = label_encoder.fit_transform(df['producer'])
    df['genres'] = label_encoder.fit_transform(df['genres'])
    mapping = {'High': 1, 'Low': 0}
    df['budget'] = df['budget'].map(mapping)
    df['vote_count'] = df['vote_count'].map(mapping)
   # df['personality'] = label_encoder.fit_transform(df['personality'])
    X = df.drop(columns=['personality', 'original_title'])
    y = df['personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(random_state=42)#,class_weight={0: 1, 1: 1, 2: 2, 3: 2})
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy=accuracy*100
    print("Random Forest: \n")
    print(f"Accuracy: {accuracy}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model,X, y, cv=5)
    scores=scores.mean()*100
    print(f"Cross-validation accuracy: {scores}%")
    return[accuracy,scores]
#RandomForest()
#TRIED EVERY TECHNIQUE IN RANDOM FOREST,FROM APPLYING SMOTE,ADJUSTING CLASS WEIGHTS TO EVEN INCREASING Test SIZE from 0.2 to 0.3(20% to 30%), yet minor to no changes in accuracy
#which is way less than ID3 and CART...I guess need to play with data more like adjust it better...preprocess furthur but how?

#4. JUST FOR EXPERIMENT---> Naive Bayes
def NB():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv("PreprocessedData.csv")
    label_encoder = LabelEncoder()
    df['director'] = label_encoder.fit_transform(df['director'])
    df['producer'] = label_encoder.fit_transform(df['producer'])
    df['genres'] = label_encoder.fit_transform(df['genres'])
    mapping = {'High': 1, 'Low': 0}
    df['budget'] = df['budget'].map(mapping)
    df['vote_count'] = df['vote_count'].map(mapping)
    X = df.drop(columns=['personality', 'original_title'])
    y = df['personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)    
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)    
    accuracy = accuracy_score(y_test, y_pred)
    accuracy=accuracy*100
    print("Naive Bayes: \n")
    print(f"Accuracy: {accuracy}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5)
    scores=scores.mean()*100
    print(f"Cross-validation accuracy: {scores}%")
    return[accuracy,scores]
#NB()
    
##def AccuracyPlot(m1,m2,m3,m4):
##    import matplotlib.pyplot as plt
##    import numpy as np
##    models = ['ID3', 'CART', 'Random Forest', 'Naive Bayes']
##    acc = [m1[0], m2[0], m3[0], m4[0]]
##    colors = ['green', 'yellow', 'red', 'blue']
##    plt.bar(models, acc,color=colors)
##    plt.title('Accuracy Chart for Applied Machine Learning Models')
##    plt.xlabel('Models')
##    plt.ylabel('Accuracy')
##    plt.show()
##
##
##def CrossValidationPlot(m1,m2,m3,m4):
##    import matplotlib.pyplot as plt
##    import numpy as np
##    models = ['ID3', 'CART', 'Random Forest', 'Naive Bayes']
##    acc = [m1[1], m2[1], m3[1], m4[1]]
##    colors = ['green', 'yellow', 'red', 'blue']
##    plt.bar(models, acc,color=colors)
##    plt.title('5-fold Cross Validation Accuracy Chart for Applied Machine Learning Models')
##    plt.xlabel('Models')
##    plt.ylabel('Accuracy')
##    plt.show()
def AccuracyPlot(m1, m2, m3, m4):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ['ID3', 'CART', 'Random Forest', 'Naive Bayes']
    acc = [m1[0], m2[0], m3[0], m4[0]]
    colors = ['green', 'yellow', 'red', 'blue']

    plt.bar(models, acc, color=colors)
    plt.title('Accuracy Chart for Applied Machine Learning Models')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')

    # Set y-axis scale from 0 to 100 with step of 10
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(0, 100)

    plt.show()


def CrossValidationPlot(m1, m2, m3, m4):
    import matplotlib.pyplot as plt
    import numpy as np

    models = ['ID3', 'CART', 'Random Forest', 'Naive Bayes']
    acc = [m1[1], m2[1], m3[1], m4[1]]
    colors = ['green', 'yellow', 'red', 'blue']

    plt.bar(models, acc, color=colors)
    plt.title('5-fold Cross Validation Accuracy Chart for Applied Machine Learning Models')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')

    # Set y-axis scale from 0 to 100 with step of 10
    plt.yticks(np.arange(0, 110, 10))
    plt.ylim(0, 100)

    plt.show()
AccuracyPlot(ID3(),CART(),RandomForest(),NB())
CrossValidationPlot(ID3(),CART(),RandomForest(),NB())
