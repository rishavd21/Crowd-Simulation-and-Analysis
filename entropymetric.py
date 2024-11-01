import orcatoimport1
import numpy as np
from orcatoimport1 import fhat
import pandas as pd

# read the data from the csv file
df = pd.read_csv(r'C:\Users\Lenovo\Downloads\red_contour_centers lesstime1.csv')
selected_columns1 = df.iloc[:, 0: 8]
selected_columns2 = df.iloc[:, 10: 18]
measured_data1 = selected_columns1.to_numpy()
measured_data2 = selected_columns2.to_numpy()
measured_data = []
measured_data.append(measured_data1)
measured_data.append(measured_data2)
measured_data = np.array(measured_data)

# Transpose the array to shape (t, 2, 8)
measured_data = np.transpose(measured_data, (1, 0, 2))

#  IMPORTANT
# indices corrosponding to the current position and current velocity, they set up the variance matrix
# these indices are to are to be adjusted according to the number of objects and what you want to add in the variance matrix
# also see the comments in the ensemble_kalman_filter
indices = [1, 2, 6, 7, 9, 10, 14, 15]


def h(state):
    
    return state[indices]

def ensemble_kalman_filter(z, fhat, M, Q, xini, m,jitter = 1e-6):
   
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    t = len(z)
    _, n, d = xini.shape
    xhat = np.zeros((t, m, n*d))
    xini1 = xini.reshape(m,n*d)
    xhat[0] = xini1
    d1 = len(M[0])
    
    M_curl = np.kron(np.eye(n), M)
    mean = np.zeros(n * d1)
    for k in range(1, t):
        # Predict
        zhat = np.zeros((m, n*d1))
        for i in range(m):
            sample = np.random.multivariate_normal(mean, M_curl)
            sample_reshaped = np.reshape(sample, (-1,1))
            inp = xhat[k-1][i].reshape(n,d)
            xhat[k][i] = (fhat(inp).reshape(-1,1)).flatten() 
                                                                            #the below indices will change according to the number of people and which person is stationary
            xhat[k][i][[1,2,6,7]] += sample_reshaped.flatten()[[0,1,2,3]]   # variance is added only in the indices of the moving object as the stationary object if fixed. we can remove it by using a loop
            noise = np.random.multivariate_normal(mean, Q)
            noisereshape = np.reshape(noise,(-1,1))
            zhat[i] = h(xhat[k][i]) + noisereshape.flatten()

        zbar = (float(1/m)*np.sum(zhat[t].reshape(-1,1) for t in range(m)))
        Z_k = float(1/m) * np.sum(((zhat[t].reshape(-1,1) - zbar) @ ((zhat[t].reshape(-1,1) - zbar).T)) for t in range(m))
        Z_k += jitter * np.eye(Z_k.shape[0])
        # Correct
        for j in range(1, k):
            xbar_j = float(1/m)*np.sum(xhat[j][t][indices].reshape(-1,1) for t in range(m))
            Sigma_j = float(1/m) * np.sum(((xhat[j][t][indices].reshape(-1,1) - xbar_j) @ ((zhat[t].reshape(-1,1) - zbar).T)) for t in range(m))
            
            for i in range(m):
                xhat[j][i][indices] = xhat[j][i][indices] + (Sigma_j @ np.linalg.inv(Z_k) @ (z[k].reshape(-1,1)[indices] - zhat[i].reshape(-1,1))).flatten()
                
    xhat1 = xhat.reshape(t,m,n,d)
    return xhat1





def estimate_error_variance(X, fhat):
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    X = np.array(X)
    t, m, n, d = X.shape
    
    d1 = 4 
    M = np.zeros((d1, d1))

    for k in range(t - 2):
        indices1 = [1,2,6,7]
        M_k = np.zeros((d1, d1))  # Initialize M_k for the current time step
        for i in range(m):
            # Predicted state at time step k and next time step
            x_k = X[k][i]
            x_kplus1 = X[k + 1][i]
            p = fhat(x_k)

            for j in range(n):

                diff = (x_kplus1[j] - p[j]).reshape(-1,1)[indices1]
                # print(diff)
                M_k += (diff @ diff.T)

        M += M_k 
    M /= ((t - 1)*m*n)
    sqrt_abs = np.sqrt(np.abs(M))
    return sqrt_abs


def run_cyclic_ensemble_kalman_filter(measured_data, initial_states, Q, m=10, convergence_threshold=1e-5, max_iterations=100):
    _,d = initial_states.shape
    d1= 4
    xini = np.array([initial_states.copy() for _ in range(m)])
    M = np.random.uniform(0, .2, (d1, d1))
    print(M)
    previous_M = np.zeros_like(M)
    iterations = 0

    entropy_metrics = []

    while np.linalg.norm(M - previous_M) > convergence_threshold and iterations < max_iterations:
        iterations += 1
        previous_M = M.copy()

        estimated_states = ensemble_kalman_filter(measured_data, fhat, M, Q, xini, m)
        M = estimate_error_variance(estimated_states, fhat)
        M1 = M / 1
        determinant_M = abs(np.linalg.det(M1))

        # Calculate the entropy metric
        entropy_metric = float(0.5 * np.log(((2 * np.pi * np.e) ** d1) * (determinant_M)))
        entropy_metrics.append(entropy_metric)
        print("Entropy metric after iteration", iterations, "=", entropy_metric)

        # Print the average of the last 8 entropy metrics if available
        if len(entropy_metrics) >= 15:
            avg_last_8 = sum(entropy_metrics[-15:]) / 15
            print("Average of the last 15 entropy metrics:", avg_last_8)


initial_states =np.array([
    [0.9,38.2,9.3,4.6,9.4,0.6,-0.3,0],   # radius, current position(x ,y ), final position(x1, y1), optimal speed, current speed(a, b)
    [0.9, 20, 9, 20, 9, 0, 0, 0]
    
])

Q = 0.1*np.eye(8)

estimated_states, final_M = run_cyclic_ensemble_kalman_filter(measured_data, initial_states, Q, 30)
print("Final Estimated Error Variance Matrix M:")
print(final_M)
