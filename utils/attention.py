from torch.nn import Module, Linear, Softmax
import torch
from utils.utils import length_to_mask

class Attention(Module):
    """
    Wrapper for all Attention classes.
    Classes that inherit from this, should just implement the score function
    which should return the attention weights
    """
    def __init__(self):
        super(Attention,self).__init__()

    def forward(self, value, key=None, query=None, mask=None):
        #TODO: the key, value, query is the most general formulation of attention, 
        # but maybe it's easier to understand if we use problem specific formulation
        """
        Args:
            value : [batch_size, seq_len1, hidden_dim]
            key : [batch_size, seq_len2, hidden_dim]
            query : [batch_size, seq_len1, hidden_dim]
            mask : [batch_size, seq_len1, seq_len2] or [batch_size, seq_len1, 1]
        Returns:
            context : [batch_size, seq_len2, hidden_dim]
            attention_weights : [batch_size, seq_len1, seq_len2]
        """

        if query is None:
            query = value

        if key is None:
            key = query

        attention_weights = self.score(query, key, mask) # [batch_size, seq_len2, seq_len1]
        context = attention_weights.matmul(value) #[batch_size, seq_len2, hidden_dim]

        return context, attention_weights


class GeneralAttention(Attention):
    
    def __init__(self, hidden_dim):
        super(GeneralAttention,self).__init__()

        self.hidden_dim = hidden_dim
        self.W = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.softmax = Softmax(dim=-1)

    def score(self, query, key, mask=None):
        """
        Calculate the scores as softmax(Q*W*K^T)

        Args:
            key : [batch_size, seq_len2, hidden_dim]
            query : [batch_size, seq_len1, hidden_dim]
            mask : [batch_size, seq_len1, seq_len2] or [batch_size, seq_len1, 1]
        Returns:
            attention_weights : [batch_size, seq_len1, seq_len2]
        """

        key = key.permute(0,2,1) #[batch_size, hidden_dim, seq_len2]
        attn = self.W(query).matmul(key) #[batch_size, seq_len1, seq_len2]

        # We might have to mask some of the input, if we have padding
        if mask is not None:
            attn.masked_fill_(mask,-float('inf'))

        attn = attn.permute(0,2,1) #[batch_size, seq_len2, seq_len1]

        # Softmax over the seq_len1 dimension
        attention_weights = self.softmax(attn) 

        return attention_weights


class UniformAttention(Attention):

    def __init__(self):
        super(UniformAttention, self).__init__()
        self.softmax = Softmax(dim=-1)
        

    def score(self, query, key=None, mask=None):
        """
        Calculate the scores as uniform distribution.
        This corresponds to a bag of words model
        
        Args:
            key : None
            query : [batch_size, seq_len, hidden_dim]
            mask : [batch_size, seq_len, 1] or [batch_size, seq_len]
        Returns:
            attention_weights : [batch_size, 1, seq_len]
        """

        batch_size, seq_len, hidden_dim = query.shape
        
        attn = torch.ones(batch_size, seq_len, 1, device = query.device) 
        # We might have to mask some of the input, if we have padding
        if mask is not None:
            attn.masked_fill_(mask,-float("inf"))

        attn = attn.permute(0,2,1) #[batch_size, 1, seq_len]
        # Softmax over the seq_len1 dimension
        attention_weights = self.softmax(attn) 

        return attention_weights

class ConditionalAttention(Module):
    """
    This layer corresponds to the conditional embedding in https://arxiv.org/pdf/1810.05237.pdf.
    The layer consists of an attention layer, a bag of word layer and a projection
    """
    def __init__(self, hidden_dim, use_bag_of_word=False):
        super(ConditionalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.attention = GeneralAttention(hidden_dim)
        self.bag_of_word = UniformAttention()
        self.W = Linear(in_features=hidden_dim, out_features=hidden_dim)
        # TODO: maybe it's more clear to split up the attention and bag of words?
        self.use_bag_of_word = use_bag_of_word

    def forward(self, variable, condition, variable_lengths=None, condition_lengths=None):
        """
        Args:
            variable [batch_size, var_seq_len, hidden_dim]: encoding of variable
            variable_lengts np.array[batch_size]: lengths of each sequence in the variables
            condition [batch_size, cond_seq_len, hidden_dim]: encoding of what we want to condition on
            condition_lengths np.array[batch_size]: lengths of each sequence in the condition variable
        Returns:
            H_var_cond [batch_size, hidden_dim]
        """
        # If the lengths is not given for any of the sequences, we assume that there is no mask
        if variable_lengths is None and condition_lengths is None:
            mask_var_cond = None
            mask_cond = None
        # We might not need masking for the condition, eg if seq_len=1
        elif condition_lengths is None:
            mask_var_cond = length_to_mask(variable_lengths).to(variable.device)
            mask_cond = None
        else:
            mask_var_cond = length_to_mask(variable_lengths, condition_lengths).to(variable.device) # [batch_size, var_seq_len, cond_seq_len]
            mask_cond = length_to_mask(condition_lengths).to(variable.device) # [batch_size, cond_seq_len, 1]
        # TODO: what if we only have the lengths for the condition?

        # Run attention conditioned on the column embeddings
        H_var_cond, _ = self.attention(variable, key=condition, mask=mask_var_cond) # [batch_size, num_cols_in_db, hidden_dim]

        if self.use_bag_of_word:
            # Use Bag of Words to remove column length
            H_var_cond, _ = self.bag_of_word(H_var_cond, mask=mask_cond) # [batch_size, 1, hidden_dim]
            H_var_cond = H_var_cond.squeeze(1) # [batch_size, hidden_dim]

        # Project embedding
        H_var_cond = self.W(H_var_cond) # [batch_size, num_cols_in_db, hidden_dim] or [batch_size, hidden_dim]

        return H_var_cond

class BagOfWord(Module):
    """
    Bag of words model, which take the mean of the embedding over the sequence length
    """
    def __init__(self):
        super(BagOfWord, self).__init__()
        self.attention = UniformAttention()


    def forward(self, variable, lengths):
        """
        Args:
            variable [batch_size, seq_len, hidden_dim]: embedding of the sequences
            lengths np.array[batch_size]: lengths of each sequence in the variable
        Returns:
            context [batch_size, hidden_dim]: masked mean over the sequence length
        """
        mask = length_to_mask(lengths)
        mask = mask.to(variable.device)
        # Calculate masked mean using uniform attention
        context, _ = self.attention(variable, mask=mask) # [batch_size, 1, hidden_dim]
        context = context.squeeze(1) # [batch_size, hidden_dim]
        return context

if __name__ == '__main__':
    # Simple test of attention functionality
    key = torch.rand(3,5,25)
    query = torch.rand(3,10,25)
    value = torch.rand(3,10,25)

    bow = UniformAttention()
    c, a = bow(value)

    att = GeneralAttention(hidden_dim=25)
    c,a = att(value, key, query)
