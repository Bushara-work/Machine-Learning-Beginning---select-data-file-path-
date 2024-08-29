import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.cluster import KMeans
from sklearn import linear_model
import os

def find_files(file_name, search_path):
    result = []
    for root, dirs, files in os.walk(search_path):
        if file_name in files:
            result.append(os.path.join(root, file_name))
    return result

def main():
    while True:
        #Type Pokemon.csv if the otehr file name has not been changed
        file_name = input("Enter the file name to search for: ")
        search_path = os.path.dirname(os.getcwd())
        files_found = find_files(file_name, search_path)
        
        if not files_found:
            print(f"No instances of {file_name} found.")
            choice = input("Do you want to (T)ry again or (Q)uit the program? ").strip().lower()
            if choice == 'q':
                print("Exiting the program.")
                break
            elif choice == 't':
                continue
            else:
                print("Invalid choice. Exiting the program.")
                break
        
        print(f"Found {len(files_found)} instance(s) of {file_name}:")
        for i, file_path in enumerate(files_found):
            print(f"{i + 1}. {file_path}")
        
        choice = input("Enter the number of the file you want to use: ").strip()
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files_found):
            print("Invalid choice.")
            retry_choice = input("Do you want to (T)ry again or (Q)uit the program? ").strip().lower()
            if retry_choice == 'q':
                print("Exiting the program.")
                break
            elif retry_choice == 't':
                continue
            else:
                print("Invalid choice. Exiting the program.")
                break

        global selected_file
        selected_file = files_found[int(choice) - 1]  
        print(f"You selected: {selected_file}")
        break

if __name__ == "__main__":
    main()
        
df = pd.read_csv(selected_file)


# Define colors for each Pokémon type
type_colors = {
    'Grass': '#78C850',
    'Poison': '#A040A0',
    'Fire': '#F08030',
    'Flying': '#A890F0',
    'Water': '#6890F0',
    'Bug': '#A8B820',
    'Normal': '#A8A878',
    'Electric': '#F8D030',
    'Ground': '#E0C068',
    'Fairy': '#EE99AC',
    'Fighting': '#C03028',
    'Psychic': '#F85888',
    'Rock': '#B8A038',
    'Steel': '#B8B8D0',
    'Ice': '#98D8D8',
    'Ghost': '#705898',
    'Dragon': '#7038F8',
    'Dark': '#705848'
}

# Function to get color for a Pokémon based on its type(s)
def get_color(row):
    type1 = row['Type 1']
    type2 = row['Type 2']
    return type_colors[type1], type_colors[type2] if pd.notna(type2) else None

# Define a function to perform linear regression and return the slope and intercept
def perform_linear_regression(x, y):
    return linregress(x, y)[:3]

# Function to plot the regression line with data points
def plot_regression(x, y, xlabel, ylabel):
    slope, intercept, r_value = perform_linear_regression(x, y)
    y_pred = slope * x + intercept

    print(f'slope b1 is {slope}')
    print(f'y intercept is {intercept}')
    print(f'R-squared is {r_value}')

    r_color = 'darkgreen' if r_value >= 0.95 else 'yellow' if r_value >= 0.85 else 'orange' if r_value >= 0.70 else 'red'

    plt.subplots(figsize=(6.5, 6.5))
    for i, row in df.iterrows():
        color1, color2 = get_color(row)
        plt.scatter(x[i], y[i], color=color1, edgecolor=color2, linewidth=2)

    plt.plot(x, y_pred, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(f'{ylabel} vs {xlabel}\nEquation: y = {slope:.2f}x + {intercept:.2f}', color='black', fontsize=14)
    plt.title(f'R^2 = {r_value:.2f}', color=r_color, fontsize=12)
    plt.show()

# Function to predict y value based on inputted x value
def predict_y(x_value, slope, intercept):
    return slope * x_value + intercept

# Function to plot the prediction with yellow lines
def plot_prediction(x, y, xlabel, ylabel):
    slope, intercept, r_value = perform_linear_regression(x, y)
    predicted_y = predict_y(x_value, slope, intercept)
    y_pred = slope * x + intercept

    plt.subplots(figsize=(6.5, 7))
    for i, row in df.iterrows():
        color1, color2 = get_color(row)
        plt.scatter(x[i], y[i], color=color1, edgecolor=color2, linewidth=2)

    print(f'slope b1 is {slope}')
    print(f'y intercept is {intercept}')
    print(f'R-squared is {r_value}')

    r_color = 'darkgreen' if r_value >= 0.95 else 'yellow' if r_value >= 0.85 else 'orange' if r_value >= 0.70 else 'red'

    plt.plot(x, y_pred, color='red')
    plt.axvline(x=x_value, ymin=0, ymax=(predicted_y - plt.ylim()[0]) / (plt.ylim()[1] - plt.ylim()[0]), color='orange', linestyle='--')
    plt.axhline(y=predicted_y, xmin=0, xmax=(x_value - plt.xlim()[0]) / (plt.xlim()[1] - plt.xlim()[0]), color='orange', linestyle='--')
    plt.scatter(x_value, predicted_y, color='orange', edgecolor='black', s=100, zorder=5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(f'{ylabel} vs {xlabel}\nPrediction: {predicted_y:.2f} {ylabel} for {x_value} {xlabel}\nEquation: y = {slope:.2f}x + {intercept:.2f}', color='black', fontsize=10)
    plt.title(f'R^2 = {r_value:.2f}', color=r_color, fontsize=10)
    plt.show()

    print(f'Predicted y value for x = {x_value} is {predicted_y}')

# Multiple Linear Regression
def multiple_linear_regression(X, y, X_value1, X_value2):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    #predict the y value (in this example the total) of a pokemon where the in this case Attack is 400 and the Defense is 300, and again where the Attack is 600 and the Defense is still 300:
    predicted_value = regr.predict([[X_value1, X_value2]])
    
    print('The predicted value is: ' + str(predicted_value[0][0]))
    print('The regression coefficients are: ' + str(regr.coef_[0][0]) + ' and ' + str(regr.coef_[0][1]))

#Use to check how many clusters to use in the key means clustering
def elbow_method(x,y):

    data = list(zip(x,y))

    # List to store inertia values
    inertias = []

    # Calculate inertia for different numbers of clusters
    for i in range(1, min(len(data), 10) + 1):  # Ensure n_clusters does not exceed number of samples
        kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Plot the Elbow Method
    plt.plot(range(1, min(len(data), 10) + 1), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()



#key means(average) clustering(goruping)
def key_means(x, y, clusters):
    data = list(zip(x,y))
    
    # Apply K-Means with the chosen number of clusters (e.g., 4)
    kmeans = KMeans(n_clusters=clusters, n_init='auto', random_state=42)
    kmeans.fit(data)

    # Plot the clusters
    plt.scatter(x, y, c=kmeans.labels_, cmap='viridis')
    plt.xlabel('Attack')
    plt.ylabel('Speed')
    plt.title('K-Means Clustering')
    plt.show()

#Preditct if given thing is likely to be in another class or not return the probabilty and ratio and (likely) binary result
def logistic_regression(X_column, y_column):
    # Reshape the 'Total' column for the logistic function
    X = np.array(X_column).reshape(-1, 1)

    # Convert 'Legendary' column to binary values
    binary_list = [1 if status else 0 for status in y_column]
    y = np.array(binary_list)

    # Train the logistic regression model
    logr = linear_model.LogisticRegression()
    logr.fit(X, y)

    # Predict if a Pokémon with 846 total stats is legendary
    prediction = logr.predict(np.array([446]).reshape(-1,1))
    if prediction == 0:
        print("Non-Legendary")
    else:
        print('Legendary')

    log_odds = logr.coef_
    odds = np.exp(log_odds)

    print('Ratio of occurance to non occurance is: ' + str(odds[0][0]))
    print('Thus it is ' + str(odds[0][0]) + ' times as likely to be legendary than not')

    # Calculate the probability for a Pokémon with 846 total stats
    probability = logit2prob(logr, np.array([846]).reshape(-1, 1))

    # Convert the probability to percentage and round to two decimal points
    percentage = round(probability[0][0] * 100, 2)
    print(f"Probability of being Legendary: {percentage}%")

#Calculate probabilty for the logistic regression
def logit2prob(logr, X):
    log_odds = logr.coef_ * X + logr.intercept_
    odds = np.exp(log_odds)
    probability = odds / (1 + odds)

    log_odds = logr.coef_
    odds = np.exp(log_odds)
    return probability

#Make a line that follows the data (more accurate than linear regression since it is limited)
def polynomial_regression(x, y):
    mymodel = np.poly1d(np.polyfit(x, y, 3))

    myline = np.linspace(1, 22, 100)

    plt.scatter(x, y)
    plt.plot(myline, mymodel(myline))
    plt.show()

 

# Example usage
plot_regression(df['Total'], df['Attack'], 'Total', 'Attack')
plot_regression(df['Sp. Atk'], df['Sp. Def'], 'Sp. Atk', 'Sp. Def')
plot_regression(df['Defense'], df['Speed'], 'Defense', 'Speed')

# Predict y value for a given x value
x_value = 100  # Example input
plot_prediction(df['Attack'], df['Defense'], 'Attack', 'Defense')



# Multiple Linear Regression
features = df[['Attack', 'Defense']].values
target = df[['Total']].values
X_value1 = 50
X_value2 = 50
multiple_linear_regression(features, target, X_value1, X_value2)
#or
multiple_linear_regression(df[['Attack', 'Defense']].values, df[['Total']].values, 600, 300)



# Define the data for clustering
elbow_method(df['Attack'].values, df['Speed'].values)

key_means(df['Attack'].values, df['Speed'].values, 4)



#Logistic Regression
logistic_regression(df['Total'], df['Legendary'])


#Polynomial Regression 
polynomial_regression(df['HP'], df['Total'])
polynomial_regression(df['Defense'], df['Total'])
polynomial_regression(df['Attack'], df['Total'])