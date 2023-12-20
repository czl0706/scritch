import os
import numpy as np
from sercomm import ser, ser_prepare
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

acc = np.zeros((3, 50), dtype=np.float32)

ser_prepare()

fig, ax = plt.subplots(figsize=(10, 6))
lines, = ax.plot([], [], label='X-axis')
lines2, = ax.plot([], [], label='Y-axis')
lines3, = ax.plot([], [], label='Z-axis')

ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, 50)

def init():
    lines.set_data([], [])
    lines2.set_data([], [])
    lines3.set_data([], [])
    return lines, lines2, lines3

def update(frame):
    global acc


    lines.set_data(range(50), acc[0,:])
    lines2.set_data(range(50), acc[1,:])
    lines3.set_data(range(50), acc[2,:])
    return lines, lines2, lines3

animation = FuncAnimation(fig, update, init_func=init, frames=50, interval=10, blit=True)

# Move plt.show() outside the loop
# plt.show(block=False)

while True:
    if ser.in_waiting:
        data = list(map(float, ser.readline().decode().strip().split(',')[1:4]))
        acc = np.roll(acc, -1, axis=1)
        acc[:, -1] = data
        
        # plt.draw()
