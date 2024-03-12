import numpy as np
import statsmodels.api as sm

AE = 0.1
ToTt = np.array([1, 2, 3, 4, 5])
AEt = np.array([0.2, 0.15, 0.1, 0.05, 0.0])


TS = sm.add_constant(ToTt)
model = sm.OLS(AEt, TS).fit()

residuals = model.resid
ACT = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

if ACT is not None and ACT >= 0.05:
    TSLM = sm.OLS(AEt, TS).fit()

    if TSLM.pvalues[1] < 0.05:
        if TSLM.params[1] < 0:

            predicted_value = TSLM.predict([1, -AE])[0]
            print("TimeOnTask threshold value corresponding to AE:", predicted_value)
        else:
            print("NA")
    else:
        print("NA")
else:
    print("NA")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.scatter(ToTt, AEt, label="Original Data", marker='o')

# Plot the linear regression model
plt.plot(ToTt, TSLM.predict(TS), label="Linear Model", color='red')

plt.xlabel("TimeOnTask")
plt.ylabel("Relative Engagement Change")
plt.title("TimeOnTask vs. Relative Engagement Change")
plt.legend()
plt.grid(True)
plt.show()
