import time

iron_pos = (0,0,1,1) #top left x,y and width, height
wire_pos = (0,0,1,1)

max_allowed_centre_distance = 50

t_min = 3
t_max = 7

fault_time = 1 #time to avoid false detection

class pad:
    pad_pos = (0,0,1,1)
    pad_state = "unsoldered" #needs to be updated only if not None
    state = "unsoldered"
    substate = "nothing"

    start_time = None

    def check_overlapping(self, pos1, pos2):
        centre1 = (pos1[0] + 0.5 * pos1[2], pos1[1] + 0.5 * pos1[3]) #x + 0.5 * width, y + 0.5 * height
        centre2 = (pos2[0] + 0.5 * pos2[2], pos2[1] + 0.5 * pos2[3])
        if(abs(centre1[0] - centre2[0]) < max_allowed_centre_distance and abs(centre1[1] - centre2[1]) < max_allowed_centre_distance ):
            return True
        else:
            return False


    def update_substate(self):
        if(substate == "nothing"):
            if(iron_pos is not None and self.pad_pos is not None):
                if(self.check_overlapping(iron_pos, self.pad_pos) and (wire_pos is None or (self.check_overlapping(wire_pos, self.pad_pos)))):
                    start_time = time.time
                    substate = "start"

        elif(substate == "start"):
            if((self.check_overlapping(iron_pos, self.pad_pos) and (wire_pos is None or (self.check_overlapping(wire_pos, self.pad_pos))))):
                if(fault_time < (abs(time.time - start_time))):
                    substate = "in_process"
            else:
                substate = "nothing"

        elif(substate == "in_process"): 
            if( not ((self.check_overlapping(iron_pos, self.pad_pos) and (wire_pos is None or (self.check_overlapping(wire_pos, self.pad_pospad_pos)))))):
                substate = "nothing"
        else:
            raise Exception("Unknown substate")
        

    def state_update(self):
        if state == "unsoldered":
            if(self.pad_state == state):
                self.update_substate()
            else:
                if(abs(time.time - self.start_time) < t_max and abs(time.time - self.start_time) > t_min): #success
                    state = "presoldered"
                    substate = "nothing"
                    return "presoldering successful"
        elif state == "presoldered":
            if(self.pad_state == state):
                self.update_substate()
            else:
                if(abs(time.time - self.start_time) < t_max and abs(time.time - self.start_time) > t_min): #success
                    state = "soldered"
                    substate = "nothing"
                    return "soldering successful"
        elif state == "soldered":
            return "done"
        else:
            raise Exception("Unknown state")