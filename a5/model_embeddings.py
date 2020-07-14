#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.char_embed_size = 50
        self.drop_out_rate = 0.3
        self.embed_size = embed_size
        self.vocab = vocab
        self.char_embed = nn.Embedding(len(self.vocab.char2id), self.char_embed_size, padding_idx=self.vocab.word2id['<pad>'])
        self.cnn = CNN(char_embed_size=self.char_embed_size, word_embed_size=self.embed_size, kernel_size=5)
        self.highway = Highway(word_embed_size=self.embed_size)
        self.dropout = nn.Dropout(self.drop_out_rate)
        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        X_embed = self.char_embed(input) #(sent_len, batch, max_word_length, char_emb_size)
        #print('X_embed shape:', X_embed.shape)
        X_reshape = X_embed.permute(0,1,3,2) #(sent_len, batch, char_emb_size, max_word_length)
        #print('X_reshape shape:', X_reshape.shape)
        X_conv = self.cnn(X_reshape.view(-1, X_reshape.size(2), X_reshape.size(3))) #(sent_len*batch, embed_size)
        #print('X_conv shape:', X_conv.shape)
        X_word_embed = self.highway(X_conv) #(sent_len*batch, embed_size)
        #print('X_word_embed shape:', X_word_embed.shape) 
        X_word_embed = self.dropout(X_word_embed.view(input.shape[0], input.shape[1], self.embed_size))#(sent_len, batch, embed_size)
        #print('X_word_embed shape:', X_word_embed.shape) 
        return X_word_embed

        ### END YOUR CODE
