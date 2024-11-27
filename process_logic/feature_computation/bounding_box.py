from dataclasses import dataclass

@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    def check_overlap(self, other: 'BoundingBox') -> bool:
        return not  (self.x + self.w < other.x 
                     or other.x + other.w < self.x 
                     or self.y + self.h < other.y 
                     or other.y + other.h < self.y)