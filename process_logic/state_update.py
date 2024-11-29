import time
import pyinterface_visual_feedback as visual
iron_pos = None #top left x,y and width, height
wire_pos = []

max_allowed_centre_distance = 350
feedback = visual.UserFeedback(port="COM5")

#it does not correcpond to real soldering time
#it is a time for computer thinking
t_min = 10
t_max = 60

fault_time = 5 #time to avoid false detection

class pad:
    def __init__(self):
        self.pad_pos = None
        self.pad_state = None #needs to be updated only if not None

        self.state = "unsoldered"
        self.substate = "nothing"

        self.start_time = None

    def check_overlapping(self, pos1, pos2):
        if(pos1 is None or pos2 is None):
            return False
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
            else:
                visual.feedback.idle() 
                self.start_time = None

        elif(self.substate == "start"):
            if iron_pad_overlap and wire_check:
                time_diff = ((abs(time.time() - self.start_time)))
                print("fault time check:", time_diff)
                if fault_time < time_diff:
                    self.substate = "in_process"
                    visual.feedback.start_solder()

            else:
                self.start_time = None
                self.substate = "nothing"

        elif(self.substate == "in_process"): 
           if not (iron_pad_overlap and wire_check):
                #self.start_time = None
                self.substate = "finished"
        else:
            raise Exception("Unknown substate")
        

    def state_update(self):
        if (self.pad_pos is None) or (self.pad_state is None):
            return "no pad detected"
        print("pad_state: ", self.pad_state)
        print("pad_pos: ", self.pad_pos)
        print("state: ", self.state)
        print("substate: ", self.substate)

        #debug stuff
        if(self.start_time is not None):
            time_diff = ((abs(time.time() - self.start_time)))
            print("time diff:", time_diff)
        else:
            print("no start time set")

        global wire_pos
        if self.state == "unsoldered":
            wire_pos = []
            #if (self.check_overlapping(iron_pos, self.pad_pos)) or ((self.pad_state == self.state or self.pad_state == "in_process")):
            self.update_substate()
            if(self.substate == "finished"):
                print("time: ", self.start_time)
                time_diff = ((abs(time.time() - self.start_time)))
                print("enough time check:", time_diff)
                if(time_diff < t_max and time_diff > t_min): #success
                    self.state = "presoldered"
                    self.substate = "nothing"
                    self.start_time = None
                    visual.feedback.end_solder()
                    print("presoldering successful")
                    return "presoldering successful"
                else:
                    self.state = "invalid"
        elif self.state == "presoldered":#the problem is here
            #if (self.check_overlapping(iron_pos, self.pad_pos)) or ((self.pad_state == self.state or self.pad_state == "in_process")):
            self.update_substate()
            if self.substate == "finished":
                if(abs(time.time() - self.start_time) < t_max and abs(time.time() - self.start_time) > t_min): #success
                    self.state = "soldered"
                    self.substate = "nothing"
                    self.start_time = None
                    return "soldering successful"
                else:
                    visual.feedback.error() 
                    self.state = "invalid"
            else:
                print("data error")
        elif self.state == "soldered":
            return "done"
        elif self.state == "invalid":
            return "wrong soldering time! Pad is broken forever :("
        else:
            raise Exception("Unknown state")
        
class pcb:
    def __init__(self):
        self.pads = [pad(), pad()] #2 pads we are tracking

    def float_state_to_string_state(self, float_state):
        if float_state == 0.0:
            return "presoldered"
        if float_state == 1.0:
            return "soldered"
        if float_state == 4.0:
            return "unsoldered"
        if float_state == 5.0:
            return "in_process"
        raise Exception("unknown pad state")

    def update_pads(self, detected_pads):
        for dp in detected_pads:
            self.pads[dp.id].pad_pos = dp.bounding_box
            self.pads[dp.id].pad_state = self.float_state_to_string_state(dp.state)
        new_pcb_cond = all(pad.pad_state == 'unsoldered' or pad.pad_state is None for pad in self.pads) and any(pad.state != 'unsoldered' for pad in self.pads)
        
        if new_pcb_cond:
            return False #means that we already got a new pcb
        else: #the pcb is still same, so we just continue our process
            for i, p in enumerate(self.pads):
                print("state of pad ", str(i), ": ", p.state_update())
            return True

class logic_manager:
    pcbs = [] #contains all pcbs
    cur_pcb = None

    def update_pcbs(self, detected_pads):
        if self.cur_pcb == None:
            self.cur_pcb = pcb()
            self.pcbs.append(self.cur_pcb)
        if(not self.cur_pcb.update_pads(detected_pads)):
            self.cur_pcb = None
            return "new pcb detected"
        
        if self.cur_pcb.pads[0].state == "soldered" and self.cur_pcb.pads[1].state == "soldered":
            return "pcb successful"
            