#!/usr/bin/env python
import math
import struct
import subprocess

import Foundation
import PyObjCTools.AppHelper

import solver
import render

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

# Shortcut for UUIDs
U = Foundation.CBUUID.UUIDWithString_

# ObjC class to handle bluetooth messages and convert them into a normal cube
class WeilongAIHandler:
    def __init__(self):
        self.angle_base = [1, 0, 0, 0]
        self.reset()
        self.render()

    def reset(self):
        self.state = [0] * 6
        self.turns = [0] * 6
        self.cube = solver.Cube()

    def render(self):
        return render.render(self.cube, self.turns)

    def centralManagerDidUpdateState_(self, manager):
        self.manager = manager
        print('Scanning...')
        # Can't scan for this service specifically? This cube is pretty wonky...
        #manager.scanForPeripheralsWithServices_options_([U('1000')], None)
        manager.scanForPeripheralsWithServices_options_(None, None)

    def centralManager_didDiscoverPeripheral_advertisementData_RSSI_(self, manager,
            peripheral, data, rssi):
        ident = peripheral.identifier()
        name = peripheral.name()
        print('Found device:', ident, name, name and 'MHC' in str(name))
        if name and 'MHC' in str(name):
            #print(dir(manager))
            self.peripheral = peripheral
            #peripheral.setDelegate_(self)
            manager.connectPeripheral_options_(peripheral, None)

    def centralManager_didConnectPeripheral_(self, manager, peripheral):
        print('Connected to', repr(peripheral.UUID()))
        peripheral.setDelegate_(self)
        self.peripheral.discoverServices_([])

    def peripheral_didDiscoverServices_(self, peripheral, services):
        self.service = self.peripheral.services()[1]
        assert str(self.service.UUID()) == '1000'
        # Subscribe to 1002-1005. 1001 is a weird bogus service you can't
        # subscribe to? Only using 1003/1004 now anyways
        chars = [U(str(s)) for s in range(1002, 1006)]
        self.peripheral.discoverCharacteristics_forService_(chars, self.service)

    def peripheral_didDiscoverCharacteristicsForService_error_(self, peripheral,
            service, error):
        if error:
            print('ERROR', error)
            return
        for characteristic in self.service.characteristics():
            peripheral.setNotifyValue_forCharacteristic_(True, characteristic)

    # Parse an update message from the cube
    def peripheral_didUpdateValueForCharacteristic_error_(self, peripheral,
            characteristic, error):
        char = str(characteristic.UUID())
        value = characteristic.value()

        # Face turn messages
        if char == '1003':
            value = bytes(value)

            # Message format appears to be [8b length] + [32b timestamp, 8b
            # face, 8b turn] * length Apparently there's not real start/end
            # turn messages? Guess that makes sense since the core doesn't know
            # where the pieces actually are
            for m in range(value[0]):
                start = 1 + m*6
                packet = value[start:start+6]
                face = packet[4]
                turn = packet[5]
                turn = {220: -1, 36: 1}[turn]

                # Add up partial turn info. Since the start/end message stuff
                # is a bit weird and either unreliable or not understood (or
                # both), use a counter to determine when a turn starts/ends
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

        # Gyroscope rotation messages
        elif char == '1004':
            [_, *values] = struct.unpack('<Iffff', value)
            # Fix up quaternion values. Not totally sure why this is necessary:
            # the rotations
            [w, x, y, z] = values
            values = [w, x, z, -y]

            # Convert quaternions to rotation matrix and rotate
            v = quat_mul(self.angle_base, values)
            matrix = quat_matrix(quat_normalize(v))
            render.reset()
            render.glMultTransposeMatrixf(matrix)

            # If the user has reset, save the rotation to recalibrate
            if self.render():
                self.reset()
                self.angle_base = quat_invert(values)


if __name__ == '__main__':
    render.setup()

    import objc
    objc.setVerbose(1)

    manager = Foundation.CBCentralManager.alloc()
    manager.initWithDelegate_queue_options_(WeilongAIHandler(), None, None)

    PyObjCTools.AppHelper.runConsoleEventLoop()
