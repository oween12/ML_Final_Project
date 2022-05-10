# ML_Final_Project

This is the Machine Learning final project git repository of Owen Bianchi and Robert Seney.

preprocessing.py - This is the file that parses the original dataset.  It will perform an 80/10/10 train/dev/test split on the data and store the files as .npy.  It will create another dataset of artifial scarring.  It will also normalize the images.
model.py - This is the file that creates the FER_CNN model, the facial emotion recognition convolutional neural network.  Here we define the different layers of the model and the forward() function.  Then we have the main function which actually trains the model.  At the top of main, one should specify the constant variables tailored towards where they store the data and where they want to save the graph and model.  Main will compute the train, dev, and test accuracies, outputting them every 100 steps.  It then calls generate_plot() and saves the model.
accuracies.py - This is a utility function that will compute the % accuracies and entropy of the models predictions compared to actual labels.
generate_plot.py - This is a utility function that will save the graph of the accuracies vs entropy.
real_time.py - This is the file that will run the real time video classifier.
haarcascade_frontalface_default.xml - haar cascade classifier
best_model.pt - trained model on original dataset
best_model_scar.pt - trained model on scar dataset
best_model_acc.png - graph of accuracies vs. epochs for original data on best_model
best_model_scar_acc.png - graph of accuracies vs. epochs for scar data on best_model_scar

In order to run these files, one must first download the dataset from Kaggle at the following link : https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data.  icml_face_data.csv is the name of the file, which we rename to raw_images.csv.  We then run the preprocessing.py file in the command line "python3 prepcrocessing.py".  Note that one may need to change the way the files save and store on their personal computers.  Next is to run the model.py file "python3 model.py".  One should make sure to set the constant variables accordingly.  This is to train a new model. You could also use our already trained model and run "python3 real_time.py" for the real-time classification.
