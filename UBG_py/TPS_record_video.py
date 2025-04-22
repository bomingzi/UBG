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
now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
record_path = 'record/' + now_time + "/"
pic_count = 0
start_times = 0
os.mkdir(record_path)

msg = MSG()
msg.msg_type = MSGType.SEND_CTRL_CMD
msg.msg_content = cmd_event([1, 0, 0, 0]).to_bytes()
reply = req.make_request(msg)
while True:
    msg = MSG()
    msg.msg_type = MSGType.QUERY_IMG
    reply = req.make_request(msg)

    img = msg_decode_as_image_bgr(reply)
    img = (255 * (img / 255) ** 0.4).astype(np.uint8)
    cv2.imwrite(record_path + now_time + '_' + str(start_times) + ".jpg", img)
    print("record_path:", record_path + now_time + '_' + str(start_times) + ".jpg")
    start_times += 1
    time.sleep(1/30)
