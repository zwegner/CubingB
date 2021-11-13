# CubingB, copyright 2021 Zach Wegner
#
# This file is part of CubingB.
#
# CubingB is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# CubingB is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with CubingB.  If not, see <https://www.gnu.org/licenses/>.

import struct

import libdispatch
import Foundation
import objc
import PyObjCTools.AppHelper

# Shortcut for UUIDs
U = Foundation.CBUUID.UUIDWithString_

FACE_MAP = [1, 3, 5, 2, 4, 0]
TURN_MAP = {-36: -1, 36: 1}

# ObjC class to connect to a bluetooth cube and parse messages
class WeilongAIDelegate:
    def __init__(self, handler):
        self.handler = handler

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
                [ts, face, turn] = struct.unpack('>Ibb', packet)
                face = FACE_MAP[face]
                turn = TURN_MAP[turn]

                self.handler.update_turn(face, turn, ts)

        # Gyroscope rotation messages
        elif char == '1004':
            # Here's a fun tidbit about the packet format: the timestamps
            # are big endian and the quaternion floats are little endian
            [ts] = struct.unpack('>I', value[:4])
            values = struct.unpack('<ffff', value[4:])
            # Fix up quaternion values. Not totally sure why this is necessary:
            [w, x, y, z] = values
            quat = [w, x, -z, y]

            self.handler.update_rotation(quat, ts)

def init_bluetooth(handler):
    objc.setVerbose(1)

    # Get the global dispatch queue. By default events get dispatched to
    # the main thread, which is weird and irritating in our case when we
    # try to have the render thread wait for events
    queue = libdispatch.dispatch_get_global_queue(
            libdispatch.DISPATCH_QUEUE_PRIORITY_DEFAULT, 0)

    manager = Foundation.CBCentralManager.alloc()
    manager.initWithDelegate_queue_options_(WeilongAIDelegate(handler),
            queue, None)

    # Return manager so main thread can keep a reference to it
    return manager
