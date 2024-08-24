import matplotlib.pyplot as plt

# Data to plot
sizes = [15, 30, 45, 10]  # The values for each slice
labels = ['A', 'B', 'C', 'D']  # The labels for each slice

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Display the pie chart
plt.show()
