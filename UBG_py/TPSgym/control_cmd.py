# encoding:utf-8

import struct

MOVE = 1
ROTATE = 2
EVENT = 3
POSE = 4


class CTRLCMD:
    def __init__(self) -> None:
        self.cmd_type: int = 0
        self.cmd_content: list = []

    def to_bytes(self):
        if self.cmd_type == 0:
            return None
        data = b''
        data += struct.pack("i", self.cmd_type)
        for x in self.cmd_content:
            data += struct.pack("f", x)
        return data


def cmd_move(x: list):
    cmd = CTRLCMD()
    cmd.cmd_type = MOVE
    cmd.cmd_content = x
    return cmd


def cmd_rotate(x: list):
    cmd = CTRLCMD()
    cmd.cmd_type = ROTATE
    cmd.cmd_content = x
    return cmd


def cmd_event(x: list):
    cmd = CTRLCMD()
    cmd.cmd_type = EVENT
    cmd.cmd_content = x
    return cmd


def cmd_pose(x: list):
    cmd = CTRLCMD()
    cmd.cmd_type = POSE
    cmd.cmd_content = x
    return cmd
