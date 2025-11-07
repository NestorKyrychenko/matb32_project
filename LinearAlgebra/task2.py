import numpy as np
import matplotlib.pyplot as plt

# --Task 2---------------------------------


A = np.array([[1, 3, 2],  # Define Matrix "A"
              [-3, 4, 3],
              [2, 3, 1]])

z_01 = np.array([8, 3, 12])
z_02 = (1/19)/np.array([1, -12, 19])
z_03 = (1/19)*np.array([1, 12, -19])  # Defining initial value vectors

z_04 = (1/19)*np.array([1,0, 2])


def lim(z_0, iterations=100, epsilon=10**-6):
    current = z_0
    for a in range(iterations):
        prev = current
        current = np.matmul(A, prev)
        if np.linalg.norm(current - prev) <= epsilon:
            return (current)
    return (None)


def lim_norm(z_0, iterations=1000, epsilon=10**-14):
    current = z_0/np.linalg.norm(z_0)
    q = []
    for a in range(iterations):
        prev = current
        current = np.matmul(A, prev)
        current = current/(np.linalg.norm(current))
        q.append(np.matmul(np.matmul(np.transpose(current), A), current))
        if np.linalg.norm(current - prev) <= epsilon:
            return (current, q[a], a)
    return (None)

def lim_norm_q(z_0, iterations=1000, epsilon=10**-14):
    current = z_0/np.linalg.norm(z_0)
    q = []
    for a in range(iterations):
        prev = current
        current = np.matmul(A, prev)
        current = current/(np.linalg.norm(current))
        q.append(np.matmul(np.matmul(np.transpose(current), A), current))
        if np.linalg.norm(current - prev) <= epsilon and abs(q[a]-q[a-1]) <= epsilon:
            return (current, q[a], a)
    return (None)

epsilon_list = [10**-(n+1) for n in range (14)]

iterations_list_z_01 = [lim_norm(z_01, epsilon = epsilon_list[n])[2] for n in range(14)]
iterations_list_z_02 = [lim_norm(z_02, epsilon = epsilon_list[n])[2] for n in range(14)]
iterations_list_z_03 = [lim_norm(z_03, epsilon = epsilon_list[n])[2] for n in range(14)]

fig = plt.figure(figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot()

ax1.set_xscale('log')
ax1.set_xlim(0.1,10**-14)
ax1.set_title('Iterations vs Epsilon for vectors')
ax1.set_xlabel('Epsilon for vectors')
ax1.set_ylabel('Iterations')

ax1.plot(epsilon_list, iterations_list_z_01, label = r'$z_{01}$')
ax1.plot(epsilon_list, iterations_list_z_02, label = r'$z_{02}$')
ax1.plot(epsilon_list, iterations_list_z_03, label = r'$z_{03}$')
plt.legend(title=r'For the series of:', loc='upper left')

iterations_list_q_z_01 = [lim_norm_q(z_01, epsilon = epsilon_list[n])[2] for n in range(14)]
iterations_list_q_z_02 = [lim_norm_q(z_02, epsilon = epsilon_list[n])[2] for n in range(14)]
iterations_list_q_z_03 = [lim_norm_q(z_03, epsilon = epsilon_list[n])[2] for n in range(14)]

fig2 = plt.figure(figsize=plt.figaspect(0.5))
ax2 = fig2.add_subplot()

ax2.set_xscale('log')
ax2.set_xlim(0.1,10**-14)
ax2.set_title('Iterations vs Epsilon for q')
ax2.set_xlabel('Epsilon for q')
ax2.set_ylabel('Iterations')

ax2.plot(epsilon_list, iterations_list_q_z_01, label = r'$z_{01}$')
ax2.plot(epsilon_list, iterations_list_q_z_02, label = r'$z_{02}$')
ax2.plot(epsilon_list, iterations_list_q_z_03, label = r'$z_{03}$')
plt.legend(title='For the series of:', loc='upper left')

plt.show()

print(lim_norm_q(z_03))
