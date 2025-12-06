import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    attention_scores = np.dot(decoder_hidden_state.T, np.dot(W_mult, encoder_hidden_states))
    softmax_vector = softmax(attention_scores)
    attention_vector = np.dot(encoder_hidden_states, softmax_vector.T)
    
    return attention_vector

def additive_attention(
    decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec
):
    """
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)

    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    """
    dec_part = np.dot(W_add_dec, decoder_hidden_state)
    enc_part = np.dot(W_add_enc, encoder_hidden_states)
    
    dec_part_broadcasted = dec_part + np.zeros_like(enc_part)
    tanh_input = np.tanh(dec_part_broadcasted + enc_part)
    
    attention_scores = np.dot(v_add.T, tanh_input) 
    softmax_vector = softmax(attention_scores)
    
    attention_vector = np.dot(encoder_hidden_states, softmax_vector.T)
    return attention_vector
