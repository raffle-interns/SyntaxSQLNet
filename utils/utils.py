import torch
import numpy as np
import pickle
import os
import shutil
import matplotlib.pyplot as plt

def calculate_mask(lengths,max_len, batch_size, device=torch.device('cpu')):

	
    #Convert to tensor so we can compare
    lengths = torch.tensor(lengths,dtype=torch.long)

    #calculate mask
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
    
    mask = mask.unsqueeze(2) #[batch_size, max_len, 1]

    return mask

	
def length_to_mask(lengths, lengths2=None, max_len=None, max_len2=None, device=torch.device('cpu')):
    """
    Converts list of lengths to a byte mask array

    Args:
        lengths [List] : list of lengths
        lengths2 [List]: list of lengths for second sequence
        max_len [Int] : maximum length of sequence 1
        max_len2 [Int] : maximum length of sequence 2
    Returns:
        mask [batch_size, max_len, 1] or [batch_size, max_len, max_len2] : Byte mask for input sequence(s)
    """
	
	#Find max length of lengths
    max_len = max_len or int(max(lengths))
    batch_size = len(lengths)
	
    mask = calculate_mask(lengths,max_len, batch_size)

    if lengths2 is not None:
        #Find max length of lengths
        max_len2 = max_len2 or int(max(lengths2))
		
        mask2 = calculate_mask(lengths2,max_len2, batch_size)
        mask2 = mask2.expand(batch_size, max_len2, max_len)
        mask2 = mask2.permute(0,2,1)

        mask = mask.expand(batch_size, max_len, max_len2) #[batch_size, max_len, max_len2]

        mask = mask & mask2
    	
    #masked_fill_ requires the mask to be a byte (uint8) tensor
    mask = mask.byte()

    return mask

	
def pad(sentences, pad_token=0):

    #Calculat the sequence length of the batch
    lengths = [len(sentence) for sentence in sentences]
    max_len = max(lengths)

    padded = []
    for example in sentences:
        #calculate how much padding is needed for this example
        pads = max_len - len(example)
		
        #Pad with zeros
        padded.append(np.pad(example, ((0,pads)), 'constant', constant_values=pad_token))
		
    padded = np.asarray(padded)
    lengths = np.asarray(lengths)
	
    return padded, lengths

def text2int(textnum, numwords={}):
    """
    Converts string of length n with numbers written in letters into actual numbers of same length

    Args:
        textnum [Str] : string
        numwords [Dictionry]: dictonary of keywords that replaces text with numbers
    Returns:
        curstring [Str]: corrected string
    """
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fourth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'eighth':8, 'ninth':9, 'tenth':10, 'twelfth':12}
    ordinal_endings = [('ieth', 'y')]

    # TODO: this breaks joined words like right-handed. What was the purpose?
    #textnum = textnum.replace('-', ' ')

    current = result = 0
    curstring = ""
    onnumber = False
    num_tokens=len(textnum.split())
    tokens=textnum.split()
    for word in tokens:
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                if num_tokens-1 == tokens.index(word):
                    curstring += word + ""  
                else:  
                    curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring

def make_dirs(directory,name):
    try:
        os.mkdir(directory+f'/{name}_train')
        os.mkdir(directory+f'/{name}_val')
    except FileExistsError:
        shutil.rmtree(directory+f'/{name}_train')
        shutil.rmtree(directory+f'/{name}_val')
        os.mkdir(directory+f'/{name}_train')
        os.mkdir(directory+f'/{name}_val')

def save_to_dirs(directory,name,info):
    [iter_epoch, train_loss_pickle, train_acc_pickle, train_num_train_pickle, val_loss_pickle, val_acc_pickle, val_num_val_pickle] = info
    with open(directory+f'{name}_train/'+'epoch_iter.pkl','wb') as f:
        pickle.dump(iter_epoch,f)
    with open(directory+f'{name}_val/'+'epoch_iter.pkl','wb') as f:
        pickle.dump(iter_epoch,f)
    with open(directory+f'{name}_train/'+'train_loss.pkl','wb') as f:
        pickle.dump(train_loss_pickle,f)
    with open(directory+f'{name}_train/'+'train_acc.pkl','wb') as f:
        pickle.dump(train_acc_pickle,f)
    with open(directory+f'{name}_train/'+'train_num_train.pkl','wb') as f:
        pickle.dump(train_num_train_pickle,f)
    with open(directory+f'{name}_val/'+'val_loss.pkl','wb') as f:
        pickle.dump(val_loss_pickle,f)
    with open(directory+f'{name}_val/'+'val_acc.pkl','wb') as f:
        pickle.dump(val_acc_pickle,f)
    with open(directory+f'{name}_val/'+'val_num_val.pkl','wb') as f:
        pickle.dump(val_num_val_pickle,f)

def plot_from_dirs(directory,name,info):
    with open(directory+f'{name}_train/'+'epoch_iter.pkl','rb') as f:
        iter_epoch = pickle.load(f)
    with open(directory+f'{name}_train/'+'train_loss.pkl','rb') as f:
        train_loss_pickle = pickle.load(f)
    with open(directory+f'{name}_train/'+'train_acc.pkl','rb') as f:
        train_acc_pickle = pickle.load(f)
    with open(directory+f'{name}_train/'+'train_num_train.pkl','rb') as f:
        train_num_train_pickle = pickle.load(f)
    with open(directory+f'{name}_val/'+'val_loss.pkl','rb') as f:
        val_loss_pickle = pickle.load(f)
    with open(directory+f'{name}_val/'+'val_acc.pkl','rb') as f:
        val_acc_pickle = pickle.load(f)
    with open(directory+f'{name}_val/'+'val_num_val.pkl','rb') as f:
        val_num_val_pickle = pickle.load(f)
    plt.subplot(2,1,1)
    plt.plot( iter_epoch, train_loss_pickle , color='blue', label='train', linewidth=2)
    plt.plot( iter_epoch, val_loss_pickle, color='olive', label='val', linewidth=2)
    plt.ylabel('loss')
    plt.subplot(2,1,2)
    plt.plot( iter_epoch, train_acc_pickle , color='blue', label='train', linewidth=2)
    plt.plot( iter_epoch, val_acc_pickle, color='olive', label='val', linewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.show()

if __name__ == '__main__':

    lengths = [1,4,6]
    lengths2 = [2,2,1]
    length_to_mask(lengths, lengths2, device=torch.device('cpu'))
    
