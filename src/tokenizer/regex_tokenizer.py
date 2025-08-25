import regex as re
from tqdm import  tqdm
from src.tokenizer.base_tokenizer import Tokenizer, get_consecutive, merge_index

# GPT split patterns includes GPT2 and GPT4 split pattern
GPT2_split_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_split_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

  def __init__(self, pattern=None):
    """
    RegexTokenizer inherited from base class Tokenizer to
    perform regex pattern and BPE to build vocabulary
    pattern used to override the default GPT4 split pattern
    """
    super().__init__()
    self.pattern = GPT4_split_pattern if pattern is None else pattern
    self.compile_pattern = re.compile(self.pattern)
    self.special_tokens = {}
    self.inverse_special_tokens = {}
    # Here the pattern is defualt to GPT4 pattern and if another pattern is given
    # during run time it is taken and compile_pattern takes the compile regex for
    # reusing it for regex operations. The special token is to store the special
    # tokens with index and special token and inverse special token is opposite
    # to it

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256

    # Splitting text into text chunks
    text_chunks = re.findall(self.compile_pattern, text)
    # the text is matched by GPT4 pattern which treats punctuations, letters,
    # numbers, contractions into seperately

    ids = [list(ch.encode('utf-8')) for ch in text_chunks]
    # the text chunks taken by regex pattern matching and then each is splitted
    # and encoded into bytes and then taking the integer tokens of it

    # merging the most frequent pairs
    merge = {}
    vocab = {index: bytes([index]) for index in range(256)}
    # Here the vocab is mapped from index -> bytes
    for i in tqdm(range(num_merges), total=num_merges):
      # to count number of times every consecutive pari
      consecutive = {}
      for chunk in ids:
        # taking get consecutive function with stats to update the pairs
        get_consecutive(chunk, consecutive)
        # Here unlike basictokenizer, the text is in form of chunks like splits
        # of text into bytes in list and then list of bytes that gives token int
        # thus it is list of list like, hello PC is [100,101,102,102,103] is for
        # hello and [32] for space [78,56] for PC, it is list of list so when
        # stats is given then it will store the each list pair in same dict but
        # without that each list will have each stats. The stats is given to
        # avoid that in the loop
      pair = max(consecutive, key=consecutive.get)
      # the maximum pair is taken for merging
      index = 256 + i
      # New index to replace is given starting from 256
      ids = [merge_index(chunk, pair, index) for chunk in ids]
      # saving it to merge dict
      merge[pair] = index
      vocab[index] = vocab[pair[0]] + vocab[pair[1]]
      # The vocab dict is mapped from index to pair, which the pair is bytes of
      # the pair numbers unpacked and added to vocabulary
      if verbose:
        print(f"merge {i+1}/ step{num_merges}: {pair} -> {index} ({vocab[index]}) had {consecutive[pair]} occurences")

    self.merges = merge
    self.vocab = vocab
    # The merge is using in encode()
    # The vocab is using in decode

  def register_special_tokens(self, special_tokens):
    """
    To add special tokens into vocabulary
    like {'<startoftext>': 1024}
    """
    self.special_tokens = special_tokens
    self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    # the special tokens is added using the functions and inverse special tokens
    # is taken opposite of special tokens with index: special token

  def decode(self, lst_ids):
    """
    The decode function converts the list of integer into the python
    strings. Like b'a' into a
    """
    part_bytes = []
    for idx in lst_ids:
      if idx in self.vocab:
        part_bytes.append(self.vocab[idx])
      elif idx in self.inverse_special_tokens:
        part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
      else:
        raise ValueError(f'Invalid token id: {idx}')

    text_bytes = b"".join(part_bytes)
    text = text_bytes.decode('utf-8', errors='replace')
    return text
    # here the part bytes is list which to store the bytes sequentially led to
    # full bytes of the word. if index is found in vocab then the bytes from vocab
    # is taken to part_bytes list and if not in vocab and in inverse special tokens
    # the special tokens like <startoftext> is fetched and it is encoded into bytes
    # and appending to list and else raise value error. The text bytes is taken by
    # forming the whole bytes and then it is decoded back to python string and if
    # there no decoding occur, ? is used to replace

  def _encode_chunk(self, text_bytes):
    """
    Function to encode chunks of text which bytes into token integers
    """
    ids = list(text_bytes)
    # the list of integer token ids
    while len(ids) >= 2:
      # only checks if the len of list is greater than 2 beacuse only that the
      # pairing is possible
      consecutive = get_consecutive(ids)
      # Here only ids beacuse it recieves flattend ids list
      pair = min(consecutive, key=lambda x: self.merges.get(x, float('inf')))
      # Here min is taken like in basictokenizer as to keep the order like in
      # the vocab is built during training like min will be 256 which is first
      # added during training, if there are no more merges then result in inf in
      # the list
      if pair not in self.merges:
        break
      # If the pair is not in merge then loop breaks
      index = self.merges[pair]
      # else the new index for the merge function will be the index of pair stored
      # in the merge dictionary
      ids = merge_index(ids, pair, index)

    return ids

  def encode_ordinary(self, text):
    """
    The function does not treat special tokens from
    the text
    Returns BPE token ids
    """
    text_chunks = re.findall(self.pattern, text)
    # The text input is splitted into patterns of GPT4
    ids = []
    # List to store the token integers returned from _encode_chunk
    for chunk in text_chunks:
      chunk_bytes = chunk.encode('utf-8')
      # The chunk from text is encoded into bytes to pass into _encode_chunk function
      chunk_ids = self._encode_chunk(chunk_bytes)
      # Here the bytes is passed into _encode_chunk function and it is converted to
      # list of ids and then checks for pair if list is valid by using get_consecutive
      # method and then take the minimum pair for the correct order and then checks
      # the pair in merge dict, if found the index mapping of the pair in merge is
      # taken as new index for merge_index which returns the pair replaced with new
      # index
      ids.extend(chunk_ids)
      # The chunk ids is a list and have to flatten as a single list so using
      # extend to add to the ids list

    return ids

  def encode(self, text, allowed_special='none_raise'):
    """
    Unlike the encode_ordinary, this function handles special tokens.
    allowed_special can be 'all' | 'none' | 'none_raise' or custom set of special
    tokens. If it is none_raise, then an error is raised
    """
    special = None
    if allowed_special == 'all':
      special = self.special_tokens
    elif allowed_special == 'none':
      special = {}
    elif allowed_special == 'none_raise':
      special = {}
      assert all(token not in text for token in self.special_tokens)
    else:
      raise ValueError(
          f"allowed_special={allowed_special} not understood by the tokenizer"
      )
    # Here the encode function is capable of doing special tokens so setting
    # special dict to store special tokens that is created using register special
    # token. If the allowed_special is all then special dict contains all the
    # special tokens that is registered, if it is none then special dict is empty
    # if none_raise it is again an empty dictionary nothing to store and assert is
    # used that if the text contains special tokens it will raise assertion error
    # if any other input is given raise a value error

    if not special:
      return self.encode_ordinary(text)
    # if there is no special token or special dict is empty no need to look for
    # this function just perform encode_ordinary function which performs the text
    # as pattern and split and then taken as bytes and move to _encode_chunk from
    # where the bytes is converted to list of token ids and then takies consecutive
    # pair and count and take minimum pair and then find  index corresponding to
    # the pair set as new index for merge-index function to get BPE token ids list

    special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
    special_chunks = re.split(special_pattern, text)
    # Here the special token pattern is taken from <start>, <end> to (<start>|<end>)
    # the escape is used to escape the get matches without regex complex and () is
    # used as grouping and it is splitted into chunks using re.split with the pattern
    # like 'hello', '<start>', 'world', '<end>'

    ids = []
    for part in special_chunks:
      if part in special:
        ids.append(special[part])
      else:
        ids.extend(self.encode_ordinary(part))
    # From the special pattern chunks each is taken in order to check if it is
    # is a special token,ie, it contains in special dictionary. If yes then the
    # special token index from specail dict is appended to the ids and if not
    # ie, the normal text is encoded using encode_ordinary function and returns
    # the token ids list

    return ids