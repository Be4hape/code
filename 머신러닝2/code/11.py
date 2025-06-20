import matplotlib.pyplot as plt
import numpy as np

# k values
ks = list(range(1, 101))

# Corresponding train accuracies (in percentages)
acc = [
    77.55, 77.22, 78.68, 77.22, 76.54, 75.98, 77.10, 76.88, 77.22, 77.44,
    77.44, 79.46, 76.77, 78.68, 80.13, 79.57, 79.35, 79.46, 79.46, 80.47,
    78.79, 79.01, 79.91, 79.69, 79.35, 79.46, 79.35, 79.57, 79.35, 78.90,
    79.24, 79.01, 78.90, 79.24, 79.12, 79.12, 79.35, 79.35, 79.35, 79.35,
    79.35, 79.35, 79.35, 79.35, 79.24, 79.24, 79.01, 79.12, 78.90, 78.90,
    78.56, 78.56, 78.56, 78.56, 78.56, 78.68, 78.68, 78.68, 78.68, 78.68,
    78.45, 78.45, 78.56, 78.68, 78.00, 78.23, 78.34, 78.23, 78.45, 78.45,
    78.68, 78.56, 78.56, 78.45, 78.68, 78.56, 78.56, 78.68, 78.45, 78.56,
    78.34, 78.68, 78.34, 78.45, 78.34, 78.23, 78.23, 78.45, 78.45, 78.34,
    78.34, 78.34, 78.34, 78.34, 78.00, 78.00, 78.45, 78.45, 78.34, 78.34
]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(ks, acc,  marker='o')
plt.scatter([20], [acc[19]], s=100, marker='o')  # Highlight k=20
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Train Accuracy (%)')
plt.title('Train Accuracy vs. k in KNN\nOptimal k = 20')
plt.grid(True)
plt.show()
