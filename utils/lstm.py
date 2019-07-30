
from torch.nn import Module, LSTM
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PackedLSTM(Module):
    """
    Wrapper around LSTM, that automatically converts to packed sequence, runs the LSTM and convert back to padded sequence
    """
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False):
        super(PackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, sequence, lengths):
        """
        Args:
            sequence [batch_size, seq_len, embedding_dim]: sequences to run the lstm over
            lenghts [batch_size]: lengths of each sequence
        Returns:
            hidden_states [batch_size, seq_len, num_directions*num_hidden]: hidden states of the final layer of the lstm

        """
        batch_size, seq_len, embedding_dim = sequence.shape
        
        # Filter out any lenghts of zero, since these just acts as padding 
        mask = lengths>0
        lengths_filtered = lengths[mask]
        sequence_filtered = sequence[torch.tensor(mask)]

        # This is just copied from net_utils
        # Sort the filtered sequences based on their length and return their indicies in lengths_filtered in reverse order 
        sort_perm = np.array(sorted(range(len(lengths_filtered)),
            key=lambda k:lengths_filtered[k], reverse=True))
        
        sort_inp_len = lengths_filtered[sort_perm]

        # Get indicies of sorted lengths in lengths_filtered
        sort_perm_inv = np.argsort(sort_perm)
        if sequence.is_cuda:
            sort_perm = torch.tensor(sort_perm).long().cuda()
            sort_perm_inv = torch.tensor(sort_perm_inv).long().cuda()

        lstm_inp = pack_padded_sequence(sequence_filtered[sort_perm],
                sort_inp_len, batch_first=True)

        sort_ret_s, _ = self.lstm(lstm_inp, None)
        ret_s = pad_packed_sequence(
                sort_ret_s, batch_first=True)[0][sort_perm_inv]

        # Some of the data was filtered as padding, but we need to add it back again    
        if ret_s.size(0) != batch_size:

            # Convert mask to binary tensor
            mask = torch.tensor(mask).byte()
            tmp = torch.zeros(batch_size, seq_len, self.hidden_size + self.hidden_size*self.bidirectional).to(ret_s.device)
            tmp[mask,:,:] = ret_s
            ret_s = tmp

        # Get the final state of the lstm, corresponding to t=seq_len     
        idx = torch.LongTensor(lengths).unsqueeze(1).unsqueeze(2).expand(batch_size,1,ret_s.size(2)).to(ret_s.device)

        # For sequences of length 0, we are taking the element at index 1, but they should all be zero so no biigie
        last_state = ret_s.gather(1, torch.abs(idx-1))    
        return ret_s, last_state
