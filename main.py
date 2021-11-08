#!/usr/bin/env python
import copy
import enum
import math
import random
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

State = enum.Enum('State', 'SCRAMBLING SCRAMBLED SOLVING SOLVED')

# Class to receive parsed bluetooth messages and turn that into regular cube logic
class CubeHandler:
    def __init__(self):
        self.change_lock = threading.Condition()
        self.quat = [1, 0, 0, 0]
        self.base_quat = self.quat
        self.matrix = quat_matrix(self.quat)
        self.gen_scramble()

    def gen_scramble(self):
        self.reset()
        self.state = State.SCRAMBLING
        self.scramble = []
        self.solve_moves = []
        self.start_time = None

        last_face = None
        turns = list(solver.TURN_STR.values())
        # Just do 25 random moves for now, not random state scrambles
        for i in range(25):
            faces = [f for f in solver.FACE_STR if f != last_face]
            face = random.choice(faces)
            last_face = face
            move = face + random.choice(turns)
            self.scramble.append(move)
        self.scramble_left = self.scramble[:]
        print(' '.join(self.scramble))

    def main_loop(self):
        # Capture the return value just so it doesn't get GC'd and stop listening
        self.bt = bluetooth.init_bluetooth(handler)

        render.setup()

        while True:
            self.render()
            self.read_events()

    def render(self):
        with self.change_lock:
            self.change_lock.wait(timeout=.5)
            cube = copy.deepcopy(self.cube)
            turns = self.turns[:]
            matrix = self.matrix[:]
            scramble = self.scramble_left[:]

        render.reset()
        render.set_rotation(matrix)
        render.render_cube(cube, turns)
        render.flip()

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

    # Make a move and update any state for either a scramble or a solve
    def make_turn(self, face, turn):
        with self.change_lock:
            face = solver.FACE_STR[face]
            alg = face + solver.TURN_STR[turn]
            self.cube.run_alg(alg)
            # See if this is the next move of the scramble
            if self.state == State.SCRAMBLING:
                s_face = self.scramble_left[0][0]
                s_turn = solver.INV_TURN_STR[self.scramble_left[0][1:]]
                if face == s_face:
                    s_turn = (s_turn - turn) % 4
                    if not s_turn:
                        self.scramble_left.pop(0)
                        if not self.scramble_left:
                            print("DONE")
                            self.state = State.SCRAMBLED
                    else:
                        new_turn = solver.TURN_STR[s_turn]
                        self.scramble_left[0] = face + new_turn
                else:
                    new_turn = solver.TURN_STR[-turn % 4]
                    self.scramble_left.insert(0, face + new_turn)
                print(' '.join(self.scramble_left))
            else:
                pass

    # XXX for now we use weilong units, 1/36th turns
    def update_turn(self, face, turn):
        with self.change_lock:
            # Add up partial turns
            self.turns[face] += turn

            # 9 incremental turns make a full quarter turn
            if abs(self.turns[face]) >= 9:
                turn = self.turns[face] // 9
                self.make_turn(face, turn)

                # Zero out everything but the opposite face as a sanity
                # check. Use a threshold so that a partial turn doesn't
                # mess up later turn accounting (if the turning is choppy,
                # say, one turn might start before the last completes)
                opp = face ^ 1
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

if __name__ == '__main__':
    handler = CubeHandler()
    handler.main_loop()
