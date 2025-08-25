from  src.tokenizer.regex_tokenizer import RegexTokenizer
import os

Base_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(Base_dir, "..", "tokenizer", "tokenizer_model.model")
model_file = os.path.normpath(model_file)

tokenizer = RegexTokenizer()
tokenizer.load(model_file=model_file)
def get_vocab_size(tokenizer: RegexTokenizer) -> int:
  vocab = tokenizer.vocab
  special_tokens = tokenizer.special_tokens
  return len(vocab) + len(special_tokens)

import torch
torch.manual_seed(3647)

vocab_size = get_vocab_size(tokenizer=tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
  def __init__(self, head_size: int,
               n_embedding:int,
               dropout: float,
               block_size:int) -> None:
    super().__init__()
    self.key = nn.Linear(in_features=n_embedding, out_features=head_size, bias=False)
    self.value = nn.Linear(in_features=n_embedding, out_features=head_size, bias=False)
    self.query = nn.Linear(in_features=n_embedding, out_features=head_size, bias=False)
    self.register_buffer('trill', torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.tensor) -> torch.tensor:
    # input shape is of (batch, time-step, channels)
    # output shape is of (batch, time-step, head_size)
    _, T, _ = x.shape
    # print(T)
    k = self.key(x)
    # print(f'Shape of K:{k.shape}')
    # shape batch, time-step, head size
    q = self.query(x)
    # print(f'Shape of Q: {q.shape}')
    # Shape batch, time-step, head size
    # Computing attention scores
    # (batch, time-step, head size) @ (batch, head_size, time-step) -> (batch, time-step, time-step)
    weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    weights = weights.masked_fill(
        self.trill[:T, :T] == 0, float('-inf')
    ) # (batch, time-step, time-step)
    weights = F.softmax(weights, dim=-1) # shape batch, time-step, time-step
    weights = self.dropout(weights)
    # Performing weighted aggregation of values
    v = self.value(x)
    # shape batch, time-step, head size
    out = weights @ v
    # weights of shape batch, time-step, time-step @ batch, time-step, head size
    # -> batch, time-step, head size
    return out

class MultiHeadAttention(nn.Module):
  """
  Multiple heads of self-attention in parallel
  """
  def __init__(self, num_head:int, head_size:int,
               dropout:float, n_embedding:int, block_size: int) -> None:
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size=head_size, dropout=dropout, n_embedding=n_embedding,block_size=block_size) for _ in range(num_head)])
    # Heads, a module list which contains heads from the head class of shape batch,
    # seq, head size. Here the weights are claculated for query, keys, value
    # query and keys are performing dot product calculation to get the relation
    # btwn them. The Dot product got from this claculation between query and keys
    # of each token are masked using tril for the right traiangular part of the matrix
    # for not looking into future ones, instead just look the token of itself and previous
    self.projections = nn.Linear(in_features=head_size * num_head, out_features=n_embedding)
    # Projects the inputs to head size*num_head , to match embedding size for next
    # layer to process
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # print(f'After concatanation:{out.shape}')
    # Here the data will be shape of batch,token len, head size so after transform
    # it has to retrive back to original embedding dimension so concatanate the
    # splitted head size(embedding_dim // n_head) with n_heads
    out = self.dropout(self.projections(out))
    return out

class FeedForward(nn.Module):
  """
  Linear layer followed by a
  non-linearity
  """
  def __init__(self, n_embedding:int,
               dropout: float) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_features=n_embedding,
                  out_features= 4 * n_embedding),
        nn.ReLU(),
        nn.Linear(4*n_embedding, n_embedding),
        nn.Dropout(dropout)
    )
    # After Multiheadattention the tokens learned in form of contextually, but
    # too there can't be linearly stacked data as if so the model will get failed
    # in complex situations. So adding relu activation function signals some neurons
    # off to preserve relevant information. By multiplying n_embedding with 4 it
    # moves to large feature space for more learning and also projected backed to
    # its original shape

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.net(x)


class Block(nn.Module):
  """The transformer Block
  """

  def __init__(self, n_embedding:int, n_head:int,
               dropout: float, block_size:int) -> None:
    super().__init__()
    head_size = n_embedding // n_head
    self.self_attention = MultiHeadAttention(num_head=n_head, head_size=head_size,
                                             dropout=dropout, n_embedding=n_embedding,
                                             block_size=block_size)
    self.feedforward = FeedForward(n_embedding=n_embedding,dropout=dropout)
    self.layer_norm1 = nn.LayerNorm(n_embedding)
    self.layer_norm2 = nn.LayerNorm(n_embedding)
    # The block is where all layers are performing it specific activities
    # The multi head attention layer performs parallel operations of different
    # different tokens with head size at same time in which each head the query,
    # key are found and compared by dot producting and masking the future ones
    # and applying softmax and performing dropout and  performs weighted aggregation
    # of the values(learned information) with shape batch,token len, head size
    # Then passed to feed forward with linear and activation function to learn
    # the patterns and adjusts the weights by gradient descending while training
    # Layer Norm are used to normalize the inputs for better training for both
    # layers

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    x = x + self.self_attention(self.layer_norm1(x))
    x = x + self.feedforward(self.layer_norm2(x))
    # Here the input x is normalized by passing to layer normalization and then
    # self attention is performing for scaled datas for better training and output
    # and adds the residual part, the original input for making more contextual
    # Then the output of above is passed to second layer norm to normalize and then
    # passed to feedforward layer for better data while training and adding the
    # output from previous layer for more contextual
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size:int,
               n_embedding:int, n_head:int,
               block_size:int, n_layer:int,
               dropout:float, device:str,
               padding_token: int) -> None:

    super().__init__()

    self.padding_token = padding_token
    self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size,
                                              embedding_dim=n_embedding)
    # From the input with shape batch, seq len passed to embedding table
    # after tokenized integers to convert it into vectors for each token
    # with shape batch, seq_len, n_embedd
    self.position_embedding_table = nn.Embedding(num_embeddings=block_size,
                                                 embedding_dim=n_embedding)
    # After embedding vectorization checks for positional the orders of tokens
    # Here num_embeddings is block size so tokens indices from 0 to block_size
    # So each tokens has n_embedding space features to check which helps in look
    # the order or position of tokens which helps in contextualizing with shape
    # seq_len, embedd_dim

    self.blocks = nn.Sequential(
        *[Block(n_embedding=n_embedding, n_head=n_head, dropout=dropout, block_size=block_size) for _ in range(n_layer)]
    )
    # After the combination of embedding table and position embedding the tokens
    # it is then passed to multi head attention layer in block for looking into
    # the attention scores in which it splits in according to head size to which
    # each token can attend into every other token in parallel in head size dims
    # with each query key value pairs in each dims which looks for contexualized
    # learning and then concatanated at the end to feature back to the original
    # n_embedd. Then it masked to not look into future ones and then performs
    # softmax for probabilities and finally the value weights are aggregated which
    # gives a good contextual meaning, the attention layer inputs are normalized
    # before applying attention scores. Then combined with the residual,ie, original'
    # to preserve information. The shape (batch,seq,embedding_dim), then passed
    # to feedforward after normalized and then undergoes non-linear patterns for
    # better learning with increased feature space and then after learning projected
    # back to original space and adds up the residual.
    self.final_layer_norm = nn.LayerNorm(n_embedding)
    # The final layer norm applies Layer norm to normalize values for better
    # training
    self.final_linear_layer = nn.Linear(in_features=n_embedding,
                                        out_features=vocab_size)
    # The final layer layer projects the tokens with n_embedding features into
    # the vocab size with shape batch, vocab size
    self.apply(self.__init__weights)

  def __init__weights(self, module:nn.Module) -> None:
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    # Thw function used to initialize normalized weights for better training
    # for avoids gradient explodes. As mean is 0 the weights will centered at
    # 0, not all values identical but has small random weights nearly 0 and std
    # 0.02 which reduces spread of data which keeps reasonable good ouputs and
    # bias is zeroed beacuse as during training after transformation of weights
    # bias is added(y=mx+c), so if bias is bigger it will cause the negative values
    # shifted into positive range which results in the ReLU activation function to
    # be non-active or unfiltered. So it may led to unwanted relation to learn and
    # results in bad learning

  def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor]
              ):
    """
    Forwrad pass of the model

    Args:
        input_tokens: Tensor of token indices of shape (batch_size, seq_length)
        targets: Optional tensor of target token indices of same shape as input
        tokens

    Returns:
        Tuple of (logits, loss) where logits has shape (batch, seq_len, vocab_size)
        and loss is optional cross-entropy loss if targets are provided
        """

    device=input_tokens.device

    B, S = input_tokens.shape
    # print(B, S)
    token_embedding = self.token_embedding_table(input_tokens)
    # print(token_embedding.shape)
    positional_embedding = self.position_embedding_table(
        torch.arange(S, device=device)
    )
    # the input tokens with shape (batch,vocab) is passed to embedding table and
    # projets to embedding feature space for embedding vectorization and then passed
    # to positional embedding with num_embedding of block size as indices for word
    # order or positional embedding with shape(batch, seq, embedd_dim)
    x = token_embedding + positional_embedding
    # Both token embedding table and positional embedding table gets added to
    # result in contextual learning of token relation and position relation
    x = self.blocks(x)
    # Then passed to blocks where all the attention layer, feedforward layer,
    # residual layer took place
    x = self.final_layer_norm(x)
    # Then the ouput is passed to  layer for normalizatiin of outputs
    logits = self.final_linear_layer(x)
    # This layers out the logits for use in cross entropy which expects logits
    # with shape batch, seq, voacb_size

    if targets is None:
      loss = None
    else:
      B, S, H = logits.shape
      logits = logits.view(B*S, H)
      targets = targets.to(logits.device).view(B*S)
      loss = F.cross_entropy(logits, targets, ignore_index=self.padding_token)

    return logits, loss
    # if there is a target token is given then cross entropy(loss function) is
    # calculated which expects input with shape 2D (batch*seq, Vocab_size) and
    # targets with 1D shape(batch*seq_len)

  @torch.no_grad()
  def generate(self, input_tokens: torch.Tensor, max_new_tokens:int,
               block_size: int, temperature: float,
               top_k: Optional[int], top_p: Optional[float]
               )-> torch.Tensor:
    """
    Generate new tokens given a context

    Args:
       Starting token indices of shape (batch, seq_len)
       max_new_tokens: Number of new tokens to generate

    Returns:
      Tensor of token indices of shape (batch, seq_len+max_new_tokens)
      """

    for _ in range(max_new_tokens):
      cropped_inputs = input_tokens[:, -block_size:]
      # The cropped inputs for considering  input tokens with size block size from
      #last for the prediction of new tokens as the last block size tokens will be
      # responsible for the new tokens to predict upon only if block size greater
      # than block size
      logits, _ = self(cropped_inputs, targets=None)
      # The logits and loss is taken from forward pass and loss is not needed here
      # so ignored and considered only logits and passing the logits to the block size
      # tokens
      logits = logits[:, -1, :] / temperature
      # Considering the last seq of the logits with shape batch, seq_len, vocab_size
      # to batch, vocab_size to consider for the next token to predict
      # Adding temperature variable by dividing to control randomness in prediction

      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")

      probs = F.softmax(logits, dim=-1)
      # Getting the softmax or probabaility distribution for the last sequence with
      # shape batch, vocab_size as gives scores of the token for every other tokens
      # in vocabulary

      if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, sorted_indices,
                                                              sorted_indices_to_remove)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)

      idx_next = torch.multinomial(probs, num_samples=1)
      # The next predicted token is taken by using multinominal with 1 sample taken
      # in which it takes maximum probability as more chance to take and other scores
      # cab be less considering every token for the contexts unlike argmax by selecting
      # only the max indices of the maximum probability
      input_tokens = torch.cat((input_tokens, idx_next), dim=1)
      # Finally the input_tokens dimension 1,ie, seq_len and idx_next the next token of
      # prediction is concatanated and getting the next token

      if idx_next.item() == tokenizer.special_tokens["<|endoftext|>"]:
        break

    return input_tokens