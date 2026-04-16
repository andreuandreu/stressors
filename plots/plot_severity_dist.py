import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0
beta = 0.1

# Define Gumbel PDF
def gumbel_pdf(x, mu, beta):
    z = (x - mu) / beta
    return  (1 / beta) * np.exp(-(z + np.exp(-z))) 

# x range
x = np.linspace(0, 1, 1000)
x_neg = np.linspace(-1, 0, 1000)
# Compute absolute value of PDF
y = abs(gumbel_pdf(x, mu, beta))
y_neg = abs(gumbel_pdf(x_neg, mu, beta))[::-1]
y_norm = y / np.sum(y)  # Normalize to sum to 1

severity_sample = np.array([])
severity_sample_abs = np.array([])
for t in range(622):
    severity_sample_abs = np.append(severity_sample_abs, np.abs(np.random.gumbel(mu, beta)))
    #severity_sample = np.append(severity_sample, np.random.gumbel(mu, beta))
                

# Plot
plt.figure()
#plt.plot(x, y_norm*100)
plt.plot(x, y+y_neg, label="|Gumbel PDF| (μ=0, β=0.1)")
#plt.hist(severity_sample, bins=33, density=True, alpha=0.2, label="Sampled severities", color='orange')
plt.hist(severity_sample_abs, bins=33, density=True, alpha=0.05, label="Sampled absolute severities", color='blue')
plt.axvline(x=0.22, color='gray', linestyle='--', label="seb. threshold = 0.22")
plt.xlabel("Severity")
plt.ylabel("percentage")
plt.title("positive values of gumbel distribution")
plt.legend(frameon=False)

plt.show()