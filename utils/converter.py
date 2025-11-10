import torch

class AttnLabelConverter(object):
    """ Attention-based converter """

    def __init__(self, character, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # [GO] for LTR, [s] for EoS.
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

        self.batch_max_length = 26  # default_batch_max_length + 1 for [GO] + 1 for [s]
        self.device = device

    def encode(self, text, batch_max_length=None):
        """ convert text-label into text-index.
        input:
            text: list of string. [str1, str2, ..., strN]
            batch_max_length: maximum length of text label in the batch. 25 by default
        output:
            text: text index tensor. (batch_size, batch_max_length + 2)
            length: list of length of each text. (batch_size)
        """
        if batch_max_length is None:
            batch_max_length = self.batch_max_length
        else:
            self.batch_max_length = batch_max_length + 2 # +1 for [GO], +1 for [s]
            
        length = [len(s) + 2 for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.insert(0, '[GO]')
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        
        return batch_text.to(self.device), torch.IntTensor(length).to(self.device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
