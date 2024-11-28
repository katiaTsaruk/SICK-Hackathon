import time

iron_pos = None #top left x,y and width, height
wire_pos = []

max_allowed_centre_distance = 300

t_min = 3
t_max = 7

fault_time = 1 #time to avoid false detection

class pad:
    pad_pos = None
    pad_state = None #needs to be updated only if not None
    state = "unsoldered"
    substate = "nothing"

    start_time = None

    def check_overlapping(self, pos1, pos2):
        centre1 = (pos1.x + 0.5 * pos1.w, pos1.y + 0.5 * pos1.h) #x + 0.5 * width, y + 0.5 * height
        centre2 = (pos2.x + 0.5 * pos2.w, pos2.y + 0.5 * pos2.h)
        if(abs(centre1[0] - centre2[0]) < max_allowed_centre_distance and abs(centre1[1] - centre2[1]) < max_allowed_centre_distance ):
            return True
        else:
            return False


    def update_substate(self):
        print("start_time: ", self.start_time)
        iron_pad_overlap = iron_pos is not None and (self.check_overlapping(iron_pos, self.pad_pos))
        wire_check = not wire_pos or any(self.check_overlapping(wire, self.pad_pos) for wire in wire_pos)

        if(self.substate == "nothing"):
            if iron_pad_overlap and wire_check:
                    print("setting start time")
                    self.start_time = time.time()
                    self.substate = "start"

        elif(self.substate == "start"):
            if iron_pad_overlap and wire_check:
                if(fault_time < (abs(time.time() - self.start_time))):
                    self.substate = "in_process"
            else:
                self.substate = "nothing"

        elif(self.substate == "in_process"): 
           if iron_pad_overlap and wire_check:
                self.substate = "nothing"
        else:
            raise Exception("Unknown substate")
        

    def state_update(self):
        if self.pad_pos is None:
            return "no pad detected"
        print("pad_state: ", self.pad_state)
        print("pad_pos: ", self.pad_pos)
        print("state: ", self.state)
        print("substate: ", self.substate)
        

        global wire_pos
        #print("wire_pos: ", wire_pos)
        if self.state == "unsoldered":
            wire_pos = []
            if(self.pad_state == self.state or self.pad_state == "in_process"):
                self.update_substate()
            else:
                print("time: ", self.start_time)
                if(abs(time.time() - self.start_time) < t_max and abs(time.time() - self.start_time) > t_min): #success
                    self.state = "presoldered"
                    self.substate = "nothing"
                    print("presoldering successful")
                    return "presoldering successful"
        elif self.state == "presoldered":
            if(self.pad_state == self.state or self.pad_state == "in_process"):
                self.update_substate()
            else:
                if(abs(time.time() - self.start_time) < t_max and abs(time.time() - self.start_time) > t_min): #success
                    self.state = "soldered"
                    self.substate = "nothing"
                    return "soldering successful"
        elif self.state == "soldered":
            return "done"
        else:
            raise Exception("Unknown state")