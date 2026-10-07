from dataclasses import dataclass, field

@dataclass
class Unit:
    work_id: str
    unit_id: str
    position: int
    label: str
    text: str
    source_type: str
    source_url: str = ""
    meta: dict = field(default_factory=dict)

def visible_units(units, position):
        return [u for u in units if u.position <= position]
