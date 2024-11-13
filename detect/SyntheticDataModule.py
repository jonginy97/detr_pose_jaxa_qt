import socket
import dqrobotics

class SyntheticDataModule:

    def __init__(self):
        self.s = None
        pass

    def connect(self, addr: str = "127.0.0.1", port:int = 20023):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.connect((addr, port))
        pass

    def _send(self, msg: str):
        self.s.sendall(bytes(msg, 'utf-8'))
        pass

    def send(self, msg: str, args: str):
        self._send("{} {}".format(msg, args))
        pass

    def start_capture(self, path: str):
        self.send("start_cap:", path)
        pass

    def set_background(self, path: str):
        self.send("set_background:", path)
        pass

    def set_forceps_state(self, pose: dqrobotics.DQ, grip: float):
        pos = pose.translation().vec3()
        ori = pose.vec4()
        self.send("x0:", "{} {} {} {} {} {} {} {}".format(pos[0], pos[1], pos[2], ori[1], ori[2], ori[3], ori[0], grip))
        pass

    def set_scissors_state(self, pose: dqrobotics.DQ, grip: float):
        pos = pose.translation().vec3()
        ori = pose.vec4()
        self.send("x1:", "{} {} {} {} {} {} {} {}".format(pos[0], pos[1], pos[2], ori[1], ori[2], ori[3], ori[0], grip))
        pass

    def set_camera_frame(self, pose: dqrobotics.DQ):
        pos = pose.translation().vec3()
        ori = pose.vec4()
        self.send("cam:", "{} {} {} {} {} {} {}".format(pos[0], pos[1], pos[2], ori[1], ori[2], ori[3], ori[0]))
        pass

    def capture(self):
        self._send("cap")
        pass

    def end_capture(self):
        self._send("end_cap")
        pass
