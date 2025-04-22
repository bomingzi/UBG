import psutil
import subprocess
import time
import os

import win32con
from pynput.keyboard import Key, Controller
import jycomm
import win32gui
import ctypes
import win32process

port = 14514
env = os.environ
env["UJYCOMM_PORT"] = str(port)
UE_path = r"C:\Program Files\Epic Games\UE_5.2\Engine\Binaries\Win64\UnrealEditor.exe"
project_path = r"E:\libaihui\learn_fps_3_restore\learn_fps_3\learn_fps_3.uproject"
project_window_name = "learn_fps_3 - 虚幻编辑器"
req = jycomm.RequestSender("127.0.0.1", port=port)
kernel32=ctypes.windll.kernel32

def kill_process_by_hwnd(hwnd):
    _,pid=win32process.GetWindowThreadProcessId(hwnd)
    handle = kernel32.OpenProcess(1, False, pid)
    kernel32.TerminateProcess(handle, 0)

def get_processes():
    proc_list = list()

    def enum_windows_callback(hwnd, lparam):
        # 获取窗口标题
        title = win32gui.GetWindowText(hwnd)
        proc_list.append((hwnd, title))

    # 遍历所有窗口
    win32gui.EnumWindows(enum_windows_callback, None)
    return proc_list


def find_certain_window(name: str, proc_list):
    for hwnd, hname in proc_list:
        if hname.find(name) != -1:
            print(f"find process {hwnd}, name {hname}")
            return hwnd, hname
    return None, None


def start_ue_project():
    procs = get_processes()
    hwnd, name = find_certain_window(project_window_name, procs)
    if hwnd is not None:
        # kill ue
        kill_process_by_hwnd(hwnd)
        time.sleep(5)
    print("starting UE process")
    p = subprocess.Popen([UE_path, project_path], shell=True, env=env)
    time.sleep(30)
    # else:
    #     os.system('pkill -f ' + str(name))
    while True:
        procs = get_processes()
        hwnd, name = find_certain_window(project_window_name, procs)
        if hwnd is None:
            time.sleep(10)
            continue
        time.sleep(10)
        hwnd1, _ = find_certain_window("消息日志", procs)
        if hwnd1 is not None:
            try:
                win32gui.PostMessage(hwnd1, win32con.WM_CLOSE, 0, 0)
            except:
                pass
        # UE 拉起至前台
        print(f"find ue process {hwnd}, {name}")
        try:
            win32gui.ShowWindow(hwnd, 1)
            win32gui.SetForegroundWindow(hwnd)
        except:
            print("fuck")
            continue
        # 发送命令
        keyboard = Controller()
        time.sleep(6)
        with keyboard.pressed(Key.alt):
            keyboard.press('p')
        time.sleep(6)
        with keyboard.pressed(Key.shift_l):
            keyboard.press(Key.f1)
        break



def libaihui_kill_ue():
    procs = get_processes()
    hwnd1, _ = find_certain_window("UnrealEditor.exe", procs)
    if hwnd1 is not None:
        kill_process_by_hwnd(hwnd1)

    hwnd2, _ = find_certain_window("CrashReportClientEditor.exe", procs)
    if hwnd2 is not None:
        kill_process_by_hwnd(hwnd2)

    hwnd3, _ = find_certain_window("learn_fps_3 Crash Reporter", procs)
    if hwnd3 is not None:
        print("got reporter")
        kill_process_by_hwnd(hwnd3)

if __name__ == "__main__":
    first_start = 1
    while True:
        msg = jycomm.MSG()
        msg.msg_type = jycomm.MSGType.EXTEND_TEST
        rep = req.make_request(msg)
        proc_alive = False
        if rep is None:
            if first_start == 1:
                start_ue_project()
                first_start = 0
                del req
                req = jycomm.RequestSender("127.0.0.1", port=port)
                msg = jycomm.MSG()
                msg.msg_type = jycomm.MSGType.EXTEND_TEST
                rep = req.make_request(msg)
                if rep is not None:
                    if rep.msg_type == jycomm.MSGType.RESPONSE:
                        print("ue project is running..")
                        proc_alive = True
            else:
                print("ue project maybe is not running..")
                time.sleep(15)

                del req
                req = jycomm.RequestSender("127.0.0.1", port=port)
                print('req')
                msg = jycomm.MSG()
                print('msg')
                msg.msg_type = jycomm.MSGType.EXTEND_TEST
                print('msg type')
                rep = req.make_request(msg)
                if rep is not None:
                    if rep.msg_type == jycomm.MSGType.RESPONSE:
                        print("ue project is running..")
                        proc_alive = True
                        first_start = 0
        else:
            if rep.msg_type == jycomm.MSGType.RESPONSE:
                print("ue project is running..")
                proc_alive = True
                first_start = 0


        if not proc_alive:
            # libaihui_add
            libaihui_kill_ue()
            # libaihui_add
            print("UE project broken, restart it..")
            start_ue_project()
            first_start = 0
            del req
            req = jycomm.RequestSender("127.0.0.1", port=port)
        time.sleep(10)
