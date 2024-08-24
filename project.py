import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = 'glass_classification.csv'
data = pd.read_csv(file_path)

# Streamlit app title
st.title("Glass Classification EDA")

# Show the first few rows of the dataset
st.subheader("Dataset")
st.dataframe(data.head())

# Check for missing values in the dataset
st.subheader("Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values)

# Show basic statistics of the dataset
st.subheader("Statistics")
st.write(data.describe())

# Distribution of each feature
st.subheader("Feature Distributions")
num_cols = len(data.columns)  # Number of columns in the dataset
num_rows = (num_cols + 2) // 3  # Calculate the number of rows needed for the subplots

# Create subplots for feature distributions
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
axes = axes.flatten()

# Plot histograms for each feature
for i, col in enumerate(data.columns):
    axes[i].hist(data[col], bins=30, edgecolor='k')
    axes[i].set_title(col)

# Hide any unused subplots
for i in range(num_cols, len(axes)):
    fig.delaxes(axes[i])

# plt.tight_layout()
st.pyplot(fig)

# Pairplot of features
st.subheader("Pairplot")
fig, axes = plt.subplots(num_cols - 1, num_cols - 1, figsize=(15, 15))

# Plot scatter plots and histograms for pairwise relationships
for i in range(num_cols - 1):
    for j in range(num_cols - 1):
        if i != j:
            axes[i, j].scatter(data.iloc[:, j], data.iloc[:, i], alpha=0.5)
            if i == num_cols - 2:
                axes[i, j].set_xlabel(data.columns[j])
            if j == 0:
                axes[i, j].set_ylabel(data.columns[i])
        else:
            axes[i, j].hist(data.iloc[:, i], bins=30, edgecolor='k')
            if i == num_cols - 2:
                axes[i, j].set_xlabel(data.columns[j])
            if j == 0:
                axes[i, j].set_ylabel(data.columns[i])

plt.tight_layout()
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(data.corr(), cmap='coolwarm')
fig.colorbar(cax)
ticks = range(num_cols)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(data.columns, rotation=90)
ax.set_yticklabels(data.columns)
st.pyplot(fig)

# Boxplots of features
st.subheader("Boxplots")
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
axes = axes.flatten()

# Plot boxplots for each feature
for i, col in enumerate(data.columns):
    axes[i].boxplot(data[col])
    axes[i].set_title(col)

# Hide any unused subplots
for i in range(num_cols, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
st.pyplot(fig)

# Class distribution
st.subheader("Class Distribution - Bar Chart")
fig, ax = plt.subplots()
class_counts = data.iloc[:, -1].value_counts()
ax.bar(class_counts.index, class_counts.values)
ax.set_xlabel('Class')
ax.set_ylabel('Count')
st.pyplot(fig)

# Class distribution - Pie chart
st.subheader("Class Distribution - Pie Chart")
fig, ax = plt.subplots()
ax.pie(class_counts, labels=class_counts.index, autopct='%2f%%', startangle=180)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)


# Scatter plot of two selected features
st.subheader("Scatter Plot")
feature_x = st.selectbox("Select X-axis feature", data.columns)
feature_y = st.selectbox("Select Y-axis feature", data.columns)
fig, ax = plt.subplots()
ax.scatter(data[feature_x], data[feature_y], alpha=0.5)
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
st.pyplot(fig)

# Summary of highest and lowest values for each feature
st.subheader("Summary of Highest and Lowest Values")

summary = pd.DataFrame({
    'Feature': data.columns,
    'Min': data.min(),
    'Max': data.max()
}).reset_index(drop=True)

st.dataframe(summary)

# Select class for pie chart
selected_class = st.selectbox("Select Class for Pie Chart", data.iloc[:, -1].unique())
class_data = data[data.iloc[:, -1] == selected_class]
class_counts = class_data.iloc[:, :-1].sum().value_counts()

# Pie chart for selected class
st.subheader(f"Pie Chart for Class {selected_class}")
fig, ax = plt.subplots()
ax.pie(class_counts, labels=class_counts.index, autopct='%1f%%', startangle=90)
ax.axis('equal') 
st.pyplot(fig)