"""
printer_profiles.py
Printer build volume presets for auto-slice sizing.
"""
from typing import Dict, Tuple, NamedTuple


class PrinterProfile(NamedTuple):
    name: str
    build_x: float   # mm
    build_y: float   # mm
    build_z: float   # mm
    margin: float    # mm clearance from edge
    notes: str = ""

    @property
    def usable_x(self): return self.build_x - self.margin * 2
    @property
    def usable_y(self): return self.build_y - self.margin * 2
    @property
    def usable_z(self): return self.build_z - self.margin * 2
    @property
    def recommended_cut_size(self):
        """Largest square that fits on this printer."""
        return min(self.usable_x, self.usable_y) - 5


PROFILES: Dict[str, PrinterProfile] = {
    # Bambu Lab
    "Bambu P2S":  PrinterProfile("Bambu Lab P2S",  256, 256, 256, 5, "CoreXY, 0.4mm nozzle"),
    "Bambu X1C":  PrinterProfile("Bambu Lab X1C",  256, 256, 256, 5, "CoreXY, multi-material AMS"),
    "Bambu A1":   PrinterProfile("Bambu Lab A1",   256, 256, 256, 5, "Bed-slinger"),
    "Bambu A1m":  PrinterProfile("Bambu Lab A1 mini", 180, 180, 180, 5, "Compact bed-slinger"),

    # Prusa
    "Prusa MK4":  PrinterProfile("Prusa MK4",       250, 210, 220, 5, "Classic i3 design"),
    "Prusa XL":   PrinterProfile("Prusa XL",         360, 360, 360, 8, "Large format, 5-tool multi-material"),
    "Prusa Mini": PrinterProfile("Prusa Mini+",      180, 180, 180, 5, "Compact MINI"),

    # Creality
    "Ender 3":    PrinterProfile("Creality Ender 3", 220, 220, 250, 8, "Entry level FDM"),
    "Ender 3 V3": PrinterProfile("Ender 3 V3",       220, 220, 250, 8, "CoreXY upgrade"),
    "CR-10":      PrinterProfile("Creality CR-10",   300, 300, 400, 8, "Large format"),
    "K1 Max":     PrinterProfile("Creality K1 Max",  300, 300, 300, 8, "High speed CoreXY"),

    # Voron
    "Voron 2.4 250": PrinterProfile("Voron 2.4 (250mm)", 250, 250, 250, 8, "CoreXY, enclosed"),
    "Voron 2.4 350": PrinterProfile("Voron 2.4 (350mm)", 350, 350, 350, 8, "Large CoreXY"),
    "Voron Trident": PrinterProfile("Voron Trident",     250, 250, 250, 8, "3-lead screw bed"),

    # Other
    "AnkerMake M5": PrinterProfile("AnkerMake M5",   235, 235, 250, 8, "High speed"),
    "FlashForge Creator 3": PrinterProfile("FlashForge Creator 3", 300, 250, 200, 8, "IDEX dual extrusion"),
    "Elegoo Neptune 4": PrinterProfile("Elegoo Neptune 4", 225, 225, 265, 8, "Budget CoreXY"),
    "Custom":     PrinterProfile("Custom",           256, 256, 256, 5, "Set your own dimensions"),
}

DEFAULT_PROFILE = "Bambu P2S"


def get_profile(name: str) -> PrinterProfile:
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def profile_names() -> list:
    return list(PROFILES.keys())
