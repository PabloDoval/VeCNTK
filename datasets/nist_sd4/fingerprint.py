import os

class fingerprint:
    """Holds the metadata of each fingerprint sample"""

    def __init__(self, identifier, order, finger, gender, fingerprintClass):
        self.id = identifier
        self.order = order
        self.finger = finger
        self.gender = gender
        self.fingerprintClass = fingerprintClass

    def __repr__(self):
        return "ID: " + str(self.id) + " ORDER: " + str(self.order) + " FINGER: " + str(self.finger) + " GENDER: " + str(self.gender) + " CLASS: " + str(self.fingerprintClass)
    def __str__(self):
        return "ID: " + str(self.id) + " ORDER: " + str(self.order) + " FINGER: " + str(self.finger) + " GENDER: " + str(self.gender) + " CLASS: " + str(self.fingerprintClass)
