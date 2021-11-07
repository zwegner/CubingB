#!/usr/bin/env python
import copy
import math
import threading
import time

import pygame

import bluetooth
import render
import solver

def quat_mul(q1, q2):
    [a, b, c, d] = q1
    [w, x, y, z] = q2
    return [a*w - b*x - c*y - d*z,
        a*x + b*w + c*d - d*y,
        a*y - b*z + c*w + d*x,
        a*z + b*y - c*x + d*w]

def quat_invert(values):
    [w, x, y, z] = values
    f = 1 / (w*w + x*x + y*y + z*z)
    return [w * f, -x * f, -y * f, -z * f]

def quat_normalize(values):
    [w, x, y, z] = values
    f = 1 / (w*w + x*x + y*y + z*z) ** .5
    return [w * f, x * f, y * f, z * f]

def quat_matrix(values):
    [w, x, y, z] = values
    return [
        w*w + x*x - y*y - z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0,
        2*x*y + 2*w*z, w*w - x*x + y*y - z*z, 2*y*z - 2*w*x, 0,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, w*w - x*x - y*y + z*z, 0,
        0, 0, 0, 1,
    ]

# Class to receive parsed bluetooth messages and turn that into regular cube logic
class CubeHandler:
    def __init__(self):
        self.change_lock = threading.Condition()
        self.quat = [1, 0, 0, 0]
        self.base_quat = self.quat
        self.matrix = quat_matrix(self.quat)
        self.reset()

    def render(self):
        with self.change_lock:
            self.change_lock.wait(timeout=.5)
            cube = copy.deepcopy(self.cube)
            turns = copy.deepcopy(self.turns)
            matrix = copy.deepcopy(self.matrix)
        # Rotate by current gyro position, then render cube
        render.reset()
        render.glMultTransposeMatrixf(matrix)
        return render.render(cube, turns)

    def read_events(self):
        with self.change_lock:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                # Space: calibrate rotation
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.base_quat = quat_invert(self.quat)
                # Enter: reset cube state
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    self.reset()

    def reset(self):
        with self.change_lock:
            self.cube = solver.Cube()
            self.turns = [0] * 6

    # Set an asynchronous event so the render thread knows something's changed
    def mark_changed(self):
        with self.change_lock:
            self.change_lock.notify_all()

    # XXX for now we use weilong units, 1/36th turns
    def turn_face(self, face, turn):
        with self.change_lock:
            # Add up partial turns
            self.turns[face] += turn

            # 9 incremental turns make a full quarter turn
            if abs(self.turns[face]) >= 9:
                alg = 'ULFRBD'[face] + ["'", '', ''][turn + 1]
                alg = solver.parse_alg(alg)
                self.cube.run_alg(alg)

                # Zero out everything but the opposite face as a sanity
                # check. Use a threshold so that a partial turn doesn't
                # mess up later turn accounting (if the turning is choppy,
                # say, one turn might start before the last completes)
                opp = [5, 3, 4, 1, 2, 0][face]
                for f in range(6):
                    if f != opp and abs(self.turns[f]) > 4:
                        self.turns[f] = 0

            self.mark_changed()

    def update_rotation(self, quat):
        with self.change_lock:
            self.quat = quat
            q = quat_mul(self.base_quat, quat)
            self.matrix = quat_matrix(quat_normalize(q))

            self.mark_changed()

def main():
    handler = CubeHandler()
    # Capture the return value just so it doesn't get GC'd and stop listening
    _ = bluetooth.init_bluetooth(handler)
    render.setup()

    while True:
        handler.render()
        handler.read_events()

if __name__ == '__main__':
    main()
