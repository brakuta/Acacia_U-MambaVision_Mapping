# Example 3 — Road extraction from aerial/UAV RGB (binary, thin structures)

Roads are thin and connected; the loss and the augmentation differ from
compact objects: Lovasz-softmax (optimises IoU directly) together with CE,
rotation augmentation (roads have no preferred orientation), large crops for
continuity, and no minimum-area filter at vectorisation time.
