#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder,self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id), bias=True)
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, 
                                           padding_idx=self.target_vocab.char2id['<pad>'])

        ### END YOUR CODE
  
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        
        char_embed = self.decoderCharEmb(input) #(length, batch, char_emb)
        output, dec_hidden = self.charDecoder(char_embed, dec_hidden) #h_t:(length, batch, hidden_size)
        scores = self.char_output_projection(output) #(length, batch, self.vocab_size)
        
        return scores, dec_hidden
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        input = char_sequence[:-1] 
        target = char_sequence[1:]
        s_t, _= self.forward(input, dec_hidden) #(length, batch, self.vocab_size)
        s_t = s_t.contiguous().view(-1, s_t.shape[-1])
        target = target.contiguous().view(-1) 
        
        CrossEntropy = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction= 'sum')
        loss = CrossEntropy(s_t, target)
        return loss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        id2char = self.target_vocab.id2char
        start_index = self.target_vocab.char2id['{']
        end_char = '}'
        charDecode = []
        current_char = torch.tensor([start_index]*batch_size, dtype=torch.long, device=device).unsqueeze(0) #(length, batch)
        dec_hidden = initialStates
        
        for t in range(max_length):
            scores, (h_t,c_t) = self.forward(current_char, dec_hidden) #h_t:(1,batch, hidden)
            char_out = torch.argmax(scores, dim=-1) #take the max position for each batch. (1, batch)
            current_char = char_out
            charDecode.append([id2char[idx.item()] for idx in char_out[0]])
            
        decodedWords = []
        for i in range(batch_size):
            word = ''
            for t in range(max_length):
                if charDecode[t][i]!= end_char:
                    word += charDecode[t][i]
                else:
                    break
            decodedWords.append(word)
        return decodedWords
        ### END YOUR CODE

