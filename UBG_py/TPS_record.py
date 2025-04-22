from jycomm import RequestSender, MSG, MSGType
from jycomm import msg_decode_as_float_vector
from jycomm import msg_decode_as_string
from jycomm import msg_decode_as_image_bgr
import cv2
import time
import numpy as np
from TPSgym.control_cmd import CTRLCMD, cmd_event, cmd_move, cmd_rotate
import datetime
import os

req = RequestSender("127.0.0.1")
cv2.namedWindow("bh")

key_map = {
    ord("A"): "a",  # turn left
    ord("s"): "s",  # backward
    ord("D"): "d",  # turn right
    ord("w"): "w",  # forward
    ord("q"): "q",  # yaw
    ord("e"): "e",  # patch
    ord("j"): "j",  # screen capture
    ord("k"): "k",  # query location
    ord("l"): "l",  # reset
    ord("z"): "z",  # reset
    ord("x"): "x",  # reset
    ord("c"): "c",
    ord(" "): " ",  # jump
    ord("a"): "a",
    ord("S"): "s",
    ord("d"): "d",
    ord("W"): "w",
    ord("Q"): "q",
    ord("E"): "e",
    ord("J"): "j",
    ord("K"): "k",
    ord("L"): "l",
    ord("Z"): "z",  # reset
    ord("X"): "x",  # reset
    ord("C"): "c",
}

# [forward,backward,left,right,jump,reset,yaw,pitch]
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
record_path = 'record/' + now_time + "/"
pic_count = 0
start_times = 0
record = True
if record:
    os.mkdir(record_path)
while True:
    key = cv2.waitKey(30)
    if key == 27:
        print("qqqqq")
        break
    if not key in key_map:
        # print("fffff")
        continue
    kk = key_map[key]
    msg = MSG()
    print(kk)
    if kk == "w":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([1, 0]).to_bytes()
    elif kk == "s":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([-1, 0]).to_bytes()
    elif kk == "a":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([0, -1]).to_bytes()
    elif kk == "d":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([0, 1]).to_bytes()
    elif kk == "z":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_move([0, 0]).to_bytes()
    elif kk == "q":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_rotate([-1, 0]).to_bytes()
    elif kk == "e":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_rotate([1, 0]).to_bytes()
    elif kk == "j":
        msg.msg_type = MSGType.QUERY_IMG
    elif kk == "k":
        msg.msg_type = MSGType.QUERY_POS
    elif kk == "c":
        msg.msg_type = MSGType.QUERY_STATUS
    elif kk == "l":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_event([1, 0, 0, 0]).to_bytes()
    elif kk == " ":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_event([0, 1, 0, 0]).to_bytes()
    elif kk == "x":
        msg.msg_type = MSGType.SEND_CTRL_CMD
        msg.msg_content = cmd_event([0, 0, 1, 0]).to_bytes()
    reply = req.make_request(msg)
    if kk == 'k':
        print(msg_decode_as_float_vector(reply))
    elif kk == 'j':

        img = msg_decode_as_image_bgr(reply)
        img = (255 * (img / 255) ** 0.4).astype(np.uint8)
        cv2.imshow("bh", img)
        if record:
            cv2.imwrite(record_path + now_time + '_' + str(start_times) + ".jpg", img)
            print("record_path:", record_path + now_time + '_' + str(start_times) + ".jpg")
            start_times += 1
    elif kk == "c":
        print(msg_decode_as_float_vector(reply))
    else:
        pass
        # print(reply.msg_content)

cv2.destroyAllWindows()
