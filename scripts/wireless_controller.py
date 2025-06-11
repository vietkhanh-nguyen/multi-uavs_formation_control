import pygame
import numpy as np

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Controller name:", joystick.get_name())

def round_to_step(val, step=0.05):
    if val > 0:
        return np.floor(val / step) * step
    else:
        return np.ceil(val / step) * step

try:
    while True:
        pygame.event.pump()

        # Buttons: Square, X, Circle, Triangle
        cross = joystick.get_button(0)
        circle = joystick.get_button(1)
        triangle = joystick.get_button(2)
        square = joystick.get_button(3)
        hat_x, hat_y = joystick.get_hat(0)

        dpad_up = (hat_y == 1)
        dpad_down = (hat_y == -1)
        dpad_left = (hat_x == -1)
        dpad_right = (hat_x == 1)

        left_x = round_to_step(joystick.get_axis(0))
        left_y = round_to_step(joystick.get_axis(1))
        right_x = round_to_step(joystick.get_axis(3))  
        right_y = round_to_step(joystick.get_axis(4)) 

        print(f"Left Stick: ({left_x:.4f}, {left_y:.4f}) | "
              f"Right Stick: ({right_x:.4f}, {right_y:.4f}) | "
              f"Cross: {cross}, Circle: {circle}, Triangle: {triangle}, Square: {square} | "
              f"D-Pad: Up={dpad_up}, Down={dpad_down}, Left={dpad_left}, Right={dpad_right}")

except KeyboardInterrupt:
    print("\nExiting...")
