import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def read_data(file_path, value_name):
    """
    Read and preprocess data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    value_name (str): Name of the value column.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path, skiprows=4)
    df = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name=value_name)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    return df

def preprocess_data():
    """
    Read and preprocess all datasets.
    
    Returns:
    pd.DataFrame: Combined DataFrame of all datasets.
    """
    co2_emissions = read_data('CO2 Emission.csv', 'CO2_Emissions')
    gdp_per_capita = read_data('GDP per capita (current us).csv', 'GDP_per_Capita')
    population = read_data('Population total.csv', 'Population')
    renewable_energy = read_data('Renewable Energy.csv', 'Renewable_Energy')

    df = co2_emissions.merge(gdp_per_capita, on=['Country Name', 'Country Code', 'Year'])
    df = df.merge(renewable_energy, on=['Country Name', 'Country Code', 'Year'])
    df = df.merge(population, on=['Country Name', 'Country Code', 'Year'])
    
    df.dropna(inplace=True)
    return df

def scale_data(df, features):
    """
    Scale the data using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the features to be scaled.
    features (list): List of feature columns to be scaled.
    
    Returns:
    np.ndarray: Scaled features.
    StandardScaler: Scaler object.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    return scaled_features, scaler

def perform_kmeans_clustering(scaled_features, n_clusters=3, random_state=42):
    """
    Perform K-means clustering.
    
    Parameters:
    scaled_features (np.ndarray): Scaled feature array.
    n_clusters (int): Number of clusters.
    random_state (int): Random state for reproducibility.
    
    Returns:
    KMeans: KMeans clustering object.
    np.ndarray: Cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(scaled_features)
    return kmeans, cluster_labels

def plot_clustering_results(df, features, cluster_centers, silhouette_avg):
    """
    Plot clustering results.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the original data and cluster labels.
    features (list): List of feature columns used for clustering.
    cluster_centers (np.ndarray): Cluster centers.
    silhouette_avg (float): Silhouette score of the clustering.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='deep')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='black', label='Cluster Centers', marker='X')
    plt.title(f'Clustering of Countries based on {features[0]} and {features[1]}')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, features, color_map):
    """
    Plot correlation heatmap.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the features.
    features (list): List of feature columns to be included in the heatmap.
    color_map (str): Color map to be used for the heatmap.
    """
    correlation_matrix = df[features].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap=color_map, fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_population_pie_charts(df, countries, years):
    """
    Plot pie charts showing the proportion of the population for selected countries in specified years.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    countries (list): List of country names to be plotted.
    years (list): List of years to be plotted.
    """
    fig, axes = plt.subplots(1, len(years), figsize=(20, 6))
    
    for i, year in enumerate(years):
        year_data = df[(df['Country Name'].isin(countries)) & (df['Year'] == year)]
        total_population = year_data.groupby('Country Name')['Population'].sum()
        axes[i].pie(total_population, labels=total_population.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
        axes[i].set_title(f'Population Proportion in {year}')
    
    plt.tight_layout()
    plt.show()

def exponential_growth(x, a, b, c):
    """
    Exponential growth model.
    
    Parameters:
    x (array-like): Independent variable.
    a (float): Parameter a.
    b (float): Parameter b.
    c (float): Parameter c.
    
    Returns:
    array-like: Dependent variable.
    """
    return a * np.exp(b * (x - c))

def fit_exponential_growth(df, country, feature):
    """
    Fit exponential growth model to the data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    country (str): Country name.
    feature (str): Feature to be fitted.
    
    Returns:
    tuple: Fitted parameters and covariance matrix.
    """
    x_data = df[df['Country Name'] == country]['Year']
    y_data = df[df['Country Name'] == country][feature]
    params, covariance = curve_fit(exponential_growth, x_data, y_data, p0=[1, 0.01, 2000])
    return params, covariance

def plot_exponential_fit(x_data, y_data, x_fit, y_fit, params, covariance, feature, country):
    """
    Plot exponential fit with confidence intervals.
    
    Parameters:
    x_data (array-like): Original x data.
    y_data (array-like): Original y data.
    x_fit (array-like): Fitted x data.
    y_fit (array-like): Fitted y data.
    params (tuple): Fitted parameters.
    covariance (array-like): Covariance matrix.
    feature (str): Feature being fitted.
    country (str): Country name.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data', color='purple')
    plt.plot(x_fit, y_fit, color='blue', label='Exponential Fit')
    plt.fill_between(x_fit, y_fit - 1.96 * np.sqrt(np.diag(covariance)[1]), y_fit + 1.96 * np.sqrt(np.diag(covariance)[1]), color='red', alpha=0.3, label='Confidence Interval')
    plt.title(f'Exponential Fit for {feature} in {country}')
    plt.xlabel('Year')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def predict_future(params, future_years):
    """
    Predict future values using the exponential growth model.
    
    Parameters:
    params (tuple): Fitted parameters.
    future_years (array-like): Future years to predict.
    
    Returns:
    array-like: Predicted values.
    """
    future_predictions = exponential_growth(future_years, *params)
    return future_predictions

def plot_future_predictions(x_data, y_data, x_fit, y_fit, future_years, future_predictions, covariance, feature, country):
    """
    Plot future predictions with confidence intervals.
    
    Parameters:
    x_data (array-like): Original x data.
    y_data (array-like): Original y data.
    x_fit (array-like): Fitted x data.
    y_fit (array-like): Fitted y data.
    future_years (array-like): Future years to predict.
    future_predictions (array-like): Predicted future values.
    covariance (array-like): Covariance matrix.
    feature (str): Feature being predicted.
    country (str): Country name.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Historical Data', color='purple')
    plt.plot(x_fit, y_fit, color='blue', label='Exponential Fit')
    plt.errorbar(future_years, future_predictions, yerr=1.96 * np.sqrt(np.diag(covariance)[1]), fmt='o', color='red', label='Predictions')
    plt.fill_between(x_fit, y_fit - 1.96 * np.sqrt(np.diag(covariance)[1]), y_fit + 1.96 * np.sqrt(np.diag(covariance)[1]), color='red', alpha=0.3, label='Confidence Interval')
    plt.title(f'Predicted {feature} for {country}')
    plt.xlabel('Year')
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = preprocess_data()
    
    features = ['GDP_per_Capita', 'CO2_Emissions', 'Renewable_Energy', 'Population']
    scaled_features, scaler = scale_data(df, features)
    
    kmeans, cluster_labels = perform_kmeans_clustering(scaled_features)
    df['Cluster'] = cluster_labels
    
    silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
    print(f'Silhouette Score: {silhouette_avg}')
    
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plot_clustering_results(df, ['GDP_per_Capita', 'CO2_Emissions'], cluster_centers, silhouette_avg)
    
    # Plot heatmap with 'viridis' color map
    plot_correlation_heatmap(df, features, 'viridis')
    
    # Plot heatmap with 'plasma' color map
    plot_correlation_heatmap(df, features, 'plasma')
    
    plot_population_pie_charts(df, ['Pakistan', 'India'], [2000, 2010, 2020])
    
    params, covariance = fit_exponential_growth(df, 'Pakistan', 'CO2_Emissions')
    x_data = df[df['Country Name'] == 'Pakistan']['Year']
    y_data = df[df['Country Name'] == 'Pakistan']['CO2_Emissions']
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = exponential_growth(x_fit, *params)
    plot_exponential_fit(x_data, y_data, x_fit, y_fit, params, covariance, 'CO2 Emissions', 'Pakistan')
    
    future_years = np.array([2030, 2040, 2050])
    future_predictions = predict_future(params, future_years)
    print(f'Predicted CO2 Emissions for future years: {future_predictions}')
    
    plot_future_predictions(x_data, y_data, x_fit, y_fit, future_years, future_predictions, covariance, 'CO2 Emissions', 'Pakistan')

if __name__ == "__main__":
    main()
