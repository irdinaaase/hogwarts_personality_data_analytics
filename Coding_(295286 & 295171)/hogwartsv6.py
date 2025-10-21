import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set the working directory
working_directory = r"C:\Users\Irdina Balqis\Downloads"
os.chdir(working_directory)

# Read the CSV file
file_path = "hogwarts_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist in {working_directory}")

# Load the data
ad_data = pd.read_csv(file_path)

# Display initial data information
print("-------information -dataset ------")
print(ad_data.head())
print(ad_data.info())
print("\n------data description --------")
print(ad_data.describe())

# Data visualization
pd.crosstab(ad_data["Inasis"], ad_data["Hogwarts House"]).plot(kind='bar')
plt.tight_layout()
plt.show()

# Data evaluation
print("\n---------------------Num of Unique Values:--------------------")
print(ad_data.nunique())
object_variables = ['Hometown', 'Co-curriculum Activities', 'Inasis']
print("\n------ uniqueness of data / variables ---------------")
print(ad_data[object_variables].describe())
print("\n------ hometown with highest num. of students ---------------")
print(pd.crosstab(index=ad_data['Hometown'], columns='count').sort_values(['count'], ascending=False).head(30))

# Data encoding
personalityTraits_mapping = {"Agreeableness": 0, "Openness": 1, "Conscientiousness": 2, "Neuroticism": 3, "Extraversion": 4}
ad_data['Personality Traits'] = ad_data['Personality Traits'].map(personalityTraits_mapping)

behavioralTraits_mapping = {"Risk-Taking": 0, "Collaboration": 1, "Discipline": 2, "Independence": 3}
ad_data['Behavioural Traits'] = ad_data['Behavioural Traits'].map(behavioralTraits_mapping)

hobbies_mapping = {"Physical": 0, "Cerebral": 1, "Creative": 2, "Community activities": 3, "Collecting": 4, "Making & Tinkering": 5}
ad_data['Hobbies'] = ad_data['Hobbies'].map(hobbies_mapping)

def academicPerformance_mapping():
    mapping = {}
    for cgpa in range(200, 401):
        value = cgpa / 100
        if 3.67 <= value <= 4.00:
            mapping[value] = 0
        elif 3.00 <= value < 3.67:
            mapping[value] = 1
        elif 2.00 <= value < 3.00:
            mapping[value] = 2
        else:
            mapping[value] = 3
    return mapping

academicPerformance_mapping = academicPerformance_mapping()
ad_data['Academic Performance'] = ad_data['Academic Performance'].map(academicPerformance_mapping)

hometown_mapping = {"Kedah": 0, "Perlis": 0, "Pulau Pinang": 0, "Perak": 0, "Kelantan": 1, "Terengganu": 1, "Pahang": 1, "Kuantan": 1, "Selangor": 2, "Kuala Lumpur": 2, "Putrajaya": 2, "Negeri Sembilan": 2, "Melaka": 2, "Johor": 3, "Sabah": 4, "Sarawak":4, "Labuan":4}
ad_data['Hometown'] = ad_data['Hometown'].map(hometown_mapping)

inasis_mapping = {"MAS": 0, "MAYBANK": 0, "TNB": 0, "TRADEWINDS": 0, "PROTON": 0, "PETRONAS": 1, "SIME DARBY": 1, "TM": 1, "GRANTT": 1, "MISC": 1, "BSN": 1, "YAB": 2, "MUAMALAT": 2, "BANK RAKYAT": 3, "SME BANK": 3, "SISIRAN": 4, "TAMAN UNIVERSITI": 4, "Rumah Sendiri": 4}
ad_data['Inasis'] = ad_data['Inasis'].map(inasis_mapping)

coCurriculum_mapping = {"arts & creativity": 0, "visual & performing arts": 0,"community services": 1,"emergency response": 2,   "entrepreneurship & agricultures": 3, "leadership skills": 4,"martial arts": 5,"mechanical / survival skills": 6, "music": 7, "sports": 8, "uniform bodies": 9, "vocal arts": 10, "club": 11 }
ad_data['Co-curriculum Activities'] = ad_data['Co-curriculum Activities'].map(coCurriculum_mapping)

leadership_mapping = {"Yes": 0, "No": 1}
ad_data['Leadership Skills'] = ad_data['Leadership Skills'].map(leadership_mapping)

favouriteCuisine_mapping = {"Malay": 0, "Chinese": 1, "Indian": 2, "Western": 3, "Japanese": 4, "Korean": 5, "Thai": 6, "Exotic": 7}
ad_data['Favourite Cuisine'] = ad_data['Favourite Cuisine'].map(favouriteCuisine_mapping)

ad_data['Estimated Income'] = pd.to_numeric(ad_data['Estimated Income'], errors='coerce')

def income_mapping():
    mapping = {}
    for income in range(1929, 99999):
        if income >= 11820:
            mapping[income] = 0
        elif 5250 <= income < 11820:
            mapping[income] = 1
        elif income < 5250:
            mapping[income] = 2
    return mapping

income_mapping = income_mapping()
ad_data['Estimated Income'] = ad_data['Estimated Income'].map(income_mapping)

ad_data['Number of Best Friends on Campus'] = pd.to_numeric(ad_data['Number of Best Friends on Campus'], errors='coerce')

def numberOfFriends_mapping():
    mapping = {}
    for numFriends in range(0, 100):
        if numFriends > 5:
            mapping[numFriends] = 0
        elif 3 <= numFriends <= 5:
            mapping[numFriends] = 1
        elif numFriends < 3:
            mapping[numFriends] = 2
    return mapping

numberOfFriends_mapping = numberOfFriends_mapping()
ad_data['Number of Best Friends on Campus'] = ad_data['Number of Best Friends on Campus'].map(numberOfFriends_mapping)

print(ad_data['Faculty In UUM'].unique())
school_mapping = {"SEFB": 0, "SOC": 1, "STML": 2, "IBS": 3, "SBM": 4, "SOB": 5, "TISSA": 6, "SOL": 7, "STHEM": 8, "SOIS": 9, "SQS": 10, "SOG": 11, "SOE": 12, "SMMTC": 13}
ad_data['Faculty In UUM'] = ad_data['Faculty In UUM'].map(school_mapping)

print(ad_data['Hogwarts House'].unique())
hogwartsHouse_mapping = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
ad_data['Hogwarts House'] = ad_data['Hogwarts House'].map(hogwartsHouse_mapping)

# Splitting data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer

# Extracting Independent and Dependent Variables
X = ad_data[['Personality Traits', 'Behavioural Traits', 'Hobbies', 'Academic Performance', 'Hometown', 'Inasis', 'Co-curriculum Activities', 'Leadership Skills', 'Favourite Cuisine', 'Estimated Income', 'Number of Best Friends on Campus','Faculty In UUM']]
y = ad_data['Hogwarts House']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the imputer
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Plotting the correlation heatmap
numeric_data = ad_data[["Personality Traits", "Behavioural Traits", "Hobbies", "Academic Performance", "Hometown", "Inasis", "Co-curriculum Activities", "Leadership Skills", "Favourite Cuisine", "Estimated Income","Number of Best Friends on Campus","Faculty In UUM","Hogwarts House"]]
hm = sns.heatmap(data=numeric_data.corr(), annot=True)
plt.show()

# Data pre-processing
from sklearn.preprocessing import MinMaxScaler, label_binarize

# Libraries for respective classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7),
    "Naive Bayes": GaussianNB(),
    "MLP Neural Nets": MLPClassifier(solver='adam', activation='relu', alpha=1e-05, tol=1e-04, hidden_layer_sizes=(6,), random_state=1, max_iter=1000)
}

# Making predictions on the testing set
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    print(f"\n----------------------Result: {name} ----------------------")
    print(f"\nAccuracy score of {name} is {100 * metrics.accuracy_score(y_test, Y_pred):.2f}%")
    print(f"Metric classification report: {name} -->\n {metrics.classification_report(y_test, Y_pred)}")
    print(f"Confusion Matrix: {name} -->\n {metrics.confusion_matrix(y_test, Y_pred)}")

# Plot ROC curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
for classifier_name, classifier in classifiers.items():
    y_score = classifier.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{classifier_name} (area = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Create the GUI
import tkinter as tk
from tkinter import ttk, messagebox

def predict_class():
    try:
        # Get the input values and convert them using the mapping dictionaries
        personality = personalityTraits_mapping[personality_traits_var.get()]
        behavior = behavioralTraits_mapping[behavioral_traits_var.get()]
        hobbies = hobbies_mapping[hobbies_var.get()]
        academic = float(academic_performance_var.get())
        hometown = hometown_mapping[hometown_var.get()]
        inasis = inasis_mapping[inasis_var.get()]
        coCurriculum = coCurriculum_mapping[co_curriculum_activities_var.get()]
        leadership = leadership_mapping[leadership_skills_var.get()]
        cuisine = favouriteCuisine_mapping[favourite_cuisine_var.get()]
        income = float(estimated_income_var.get())
        friends = int(number_of_best_friends_var.get())
        faculty = school_mapping[faculty_in_uum_var.get()]

        input_data = np.array([[personality, behavior, hobbies, academic, hometown, inasis, coCurriculum,
                                leadership, cuisine, income, friends, faculty]])

               # Predict the house for each classifier
        house_names = {0: "Gryffindor", 1: "Hufflepuff", 2: "Ravenclaw", 3: "Slytherin"}
        predictions = []
        for name, clf in classifiers.items():
            predicted_class = clf.predict(input_data)[0]
            predictions.append(f"{name}: {house_names[predicted_class]}")

        result_message = "Predictions from each classifier:\n" + "\n".join(predictions)
        messagebox.showinfo("Prediction", result_message)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main window
window = tk.Tk()
window.title("Hogwarts House Predictor")

# Define variables
personality_traits_var = tk.StringVar()
behavioral_traits_var = tk.StringVar()
hobbies_var = tk.StringVar()
academic_performance_var = tk.StringVar()
hometown_var = tk.StringVar()
inasis_var = tk.StringVar()
co_curriculum_activities_var = tk.StringVar()
leadership_skills_var = tk.StringVar()
favourite_cuisine_var = tk.StringVar()
estimated_income_var = tk.StringVar()
number_of_best_friends_var = tk.StringVar()
faculty_in_uum_var = tk.StringVar()

# Create input fields
ttk.Label(window, text="Choose Your Personality Traits: ").grid(row=0, column=0)
ttk.Combobox(window, textvariable=personality_traits_var, values=list(personalityTraits_mapping.keys())).grid(row=0, column=1)

ttk.Label(window, text="Choose Your Behavior Traits: ").grid(row=1, column=0)
ttk.Combobox(window, textvariable=behavioral_traits_var, values=list(behavioralTraits_mapping.keys())).grid(row=1, column=1)

ttk.Label(window, text="Choose Your Hobbies Type: ").grid(row=2, column=0)
ttk.Combobox(window, textvariable=hobbies_var, values=list(hobbies_mapping.keys())).grid(row=2, column=1)

ttk.Label(window, text="Enter Your Academic CGPA: ").grid(row=3, column=0)
ttk.Entry(window, textvariable=academic_performance_var).grid(row=3, column=1)

ttk.Label(window, text="Choose Your Hometown: ").grid(row=4, column=0)
ttk.Combobox(window, textvariable=hometown_var, values=list(hometown_mapping.keys())).grid(row=4, column=1)

ttk.Label(window, text="Choose Your Inasis: ").grid(row=5, column=0)
ttk.Combobox(window, textvariable=inasis_var, values=list(inasis_mapping.keys())).grid(row=5, column=1)

ttk.Label(window, text="Choose Your Co-Curriculum Course: ").grid(row=6, column=0)
ttk.Combobox(window, textvariable=co_curriculum_activities_var, values=list(coCurriculum_mapping.keys())).grid(row=6, column=1)

ttk.Label(window, text="Are you active with Leadership? ").grid(row=7, column=0)
ttk.Combobox(window, textvariable=leadership_skills_var, values=list(leadership_mapping.keys())).grid(row=7, column=1)

ttk.Label(window, text="Choose Your Favourite Cuisine ").grid(row=8, column=0)
ttk.Combobox(window, textvariable=favourite_cuisine_var, values=list(favouriteCuisine_mapping.keys())).grid(row=8, column=1)

ttk.Label(window, text="Enter Your Parents' Estimated Income (RM): ").grid(row=9, column=0)
ttk.Entry(window, textvariable=estimated_income_var).grid(row=9, column=1)

ttk.Label(window, text="Enter Your UUM's Number of Best Friends: ").grid(row=10, column=0)
ttk.Entry(window, textvariable=number_of_best_friends_var).grid(row=10, column=1)

ttk.Label(window, text="Which Faculty Are You From?: ").grid(row=11, column=0)
ttk.Combobox(window, textvariable=faculty_in_uum_var, values=list(school_mapping.keys())).grid(row=11, column=1)

# Create predict button
ttk.Button(window, text="Click to Predict Your Hogwarts House", command=predict_class).grid(row=12, column=0, columnspan=2)

# Start the GUI event loop
window.mainloop()