# Classifying-news-into-different-categories
It classifies news and puts it into different fields

First run the createtrainingdata.py file. Here the file extracts articles fron online newspapers like BBC, The Guardian,
The Telegraph, and Reuters, scrap the latest news to create a training set of data already classified by category.
This script will generate a json file for each category inside the ``articles`` folder. That is, it creates the data to train the system.

Then run the training.py file which trains the system . Here, the algorithm which we are using is the SVM (Support Vector Machine) 
algorithm and implementing using python modules.

Then run the classify.py which classifies the news articles into different categories.

