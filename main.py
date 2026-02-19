print("Hello World!")

"""
    Preliminary work:
        - perform a descriptive analysis of dataset before preprocessing
        
    Task 1:
        Preproccessing:
            - apply NLP to preprocess dataset, not limiting to syntactic analysis
                - stemming, lemmatization, stop word removal, ner, pos
            - transform text into numerical representations (e.g. feature vectors)
                - get numeric values for each syntactic analysis part, then add into one singluar array in numerical form
                - https://www.geeksforgeeks.org/nlp/vectorization-techniques-in-nlp/
            - experiment with different approaches, assess impact and select most suitable approach
        
        Dataset split:
            - split into test & trainning, justify
            
        NN classifiers:
            - MLP
            - DL NN
            - train and record performance
            - tune networks from 3 hyperparameters
                - explain how each hyperparameter effects performance
                - use default values as a baseline
                - create visuals to support/explain choices
            - assess and evalute results from tuned networks, explain insights
            - choose best model, save it for demo
            - note down performance of best model on test and training data
            
    Task 2:
        - apply NLP and NLU to explore and analyse content in documents and their linked headlines
            to automatically discover underlying topics across text data
        
        Standard syntactic analysis:
            - parsing/tokenisation
            - stop word removal
            - lemmatization
            - stemming
            - => 2 text representation strategies (e.g. TD-IDF, BoW, LDA, Word vector or word embedding)
            - explore other NLP techniques where appropriate
        
        - understand models/experiments implemented, evaluate and analyse 
        - discuss the discovered topics and their link to document content, news headlines, and
            both the document headline- and document - class labels. 
        - prepare to explain what I done for this task
        - save best model using format of my choice to run in demo
    
    Demo:
        - powerpoint (max 7 slides)
            - provide documentation of design, model improvement regime, performance evaluation
                and discussion of results for both tasks
            - 15 mins (10 presentation, 5 Q&A)
            - run pre-trained and saved models on new data
"""