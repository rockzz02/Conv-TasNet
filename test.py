# import torch
# import numpy as np

# t = torch.randint(0, 5, (2, 1, 4))
# print(t)
# channel = t.size(1)
# time_step = t.size(2)

# t = t.sum(1)
# cum_sum = torch.cumsum(t, dim=1)
# print(cum_sum)
# print(cum_sum.shape)

# entry_cnt = np.arange(channel, channel*(time_step+1), channel)
# entry_cnt = torch.from_numpy(entry_cnt).type(t.type())
# print(entry_cnt)
# print(entry_cnt.shape)

# entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
# print(entry_cnt)
# print(entry_cnt.shape)

from graphviz import Digraph

def create_high_level_diagram():
    dot = Digraph(comment='TasNet High-Level Architecture')

    # Add nodes
    dot.node('A', 'Input')
    dot.node('B', 'Padding')
    dot.node('C', 'Encoder (self.encoder)')
    dot.node('D', 'cLN (self.cLN)')
    dot.node('E', 'TCN (self.TCN)')
    dot.node('F', 'DepthConv1d (within TCN)')
    dot.node('G', 'FCLayer (self.fc_layer)')
    dot.node('H', 'Mask Generation')
    dot.node('I', 'Masked Output')
    dot.node('J', 'Decoder (self.decoder)')
    dot.node('K', 'Final Output')

    # Add edges
    dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK'])

    # Render the diagram
    dot.render('TasNet_high_level_architecture', format='png', view=True)

if __name__ == "__main__":
    create_high_level_diagram()