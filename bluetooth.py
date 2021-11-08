import struct

import libdispatch
import Foundation
import objc
import PyObjCTools.AppHelper

# Shortcut for UUIDs
U = Foundation.CBUUID.UUIDWithString_

FACE_MAP = [1, 3, 5, 2, 4, 0]
TURN_MAP = {220: -1, 36: 1}

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
                face = FACE_MAP[packet[4]]
                turn = TURN_MAP[packet[5]]

                self.handler.update_turn(face, turn)

        # Gyroscope rotation messages
        elif char == '1004':
            [_, *values] = struct.unpack('<Iffff', value)
            # Fix up quaternion values. Not totally sure why this is necessary:
            [w, x, y, z] = values
            quat = [w, x, -z, y]

            self.handler.update_rotation(quat)

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
