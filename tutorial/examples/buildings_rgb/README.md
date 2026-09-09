# Example 2 — Building footprints from aerial RGB (binary)

Differences from the acacia example: smaller tiles (512) because buildings
need less context than tree crowns at 2.5 cm; a Dice term weighted 2:1 against
cross-entropy to counter the background majority (per-class CE weights are not
usable together with ignored pixels in MMSegmentation 1.2.2); 8-connected
post-processing and a larger minimum area at vectorisation time.
