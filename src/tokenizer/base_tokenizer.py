import unicodedata

# Building a function to find frequency of consecutive pairs
def get_consecutive(ids, counts=None):
  """
    The function will pick consecutive pairs for byte pair
    encoding
  Args:
    ids: the list of the indexes
    counts: count of pairs defualt to None
  """
  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]):
    # Iterating consecutive pair
    counts[pair] = counts.get(pair, 0) + 1
    # Adding pair to count dictionary with default as 0 so
    # it takes 0 instead of None and adding 1 to it counts the
    # repetatives

  return counts

def merge_index(indices, pair, new_index):

  """
    The function to replace all consecutive occurences of pair with
    the new integer token index
  Args:
    indices: the list of integers
    pair: the pair of consecutive indices to merge
    new_index: the new integer token to replace the pair
  """

  new_ids = []
  i = 0
  while i < len(indices):
    if indices[i] == pair[0] and i < len(indices) - 1 and indices[i+1] == pair[1]:
      new_ids.append(new_index)
      i += 2
    else:
      new_ids.append(indices[i])
      i += 1
  # Here inside while loop when i less than length of indices or list
  # starting i with 0 and checking if the element 0 of the list and 0th element
  # of pair tuple is equal and also iteration upto length of list - 1 in order to
  # checks before the last element pair because as after the last element there
  # is no element to go for pairing and also checks if the elememnt 1 and element
  # 1 of tuple is match then appending the list with new index token and shifting
  # the itertaion over 2 steps in order to avoid overlapping, else for this
  # condition it just append the indices tokens into the list

  return new_ids

def replace_control_characters(string: str) -> str:
  """
  Function to replace the control characters like \n, \b
  to unicode bytes to display while saving to vocab
  """

  chars = []
  for ch in string:
    if unicodedata.category(ch)[0] != 'C':
      chars.append(ch)
    else:
      chars.append(f'\\u{ord(ch):04x}')

  return ''.join(chars)
  # the chars list is initialized to store the characters
  # After iterating through string and checks if the first character is not start
  # with 'C', which represents the unicode character like Cc, Cf, Co and if it is
  # starting with C it is append to list with ord(ch) which is converted to ASCII
  # value which helps in visible while saving in .vocab and returning with text

def render_token(encoded: bytes) -> str:
  text = encoded.decode('utf-8', errors='replace')
  text = replace_control_characters(text)
  return text
  # the text is decoded into utf-8 which was bytes during encode and  then
  # replace the control characters using replace_control_characters function to
  # make the control characters to ord of ASCII value in order to visible
  # while saving

class Tokenizer:
  """
  Base class for Tokenizers from where
   they are abstracted
   """

  def __init__(self):
    self.merges = {}
    self.pattern = ''
    self.special_tokens = {}
    self.vocab = self.build_vocab()

  def train(self, text, vocab_size, verbose=False):
    raise NotImplementedError

  def encode(self, text):
    raise NotImplementedError

  def decode(self, ids):
    raise NotImplementedError

  def build_vocab(self):
    vocab = {index: bytes([index]) for index in range(256)}
    for (p0, p1), index in self.merges.items():
      vocab[index] = vocab[p0] + vocab[p1]

    for special, index in self.special_tokens.items():
      vocab[index] = special.encode('utf-8')

    return vocab
    # Initializing the merges with dict for storing the pairs after getting when
    # doing get_consective and merge function as key like (91,92) and its bytes
    # value as value in dict. Pattern is used to store regex patterns when using
    # the RegexTokenizer to store regex pattern like re.findall('GPT is cool') and
    # splits and later converted bytes and performs merge. The special token is
    # dict stores special tokens like <startoftext>, <endoftext> etc.. and it is
    # assigned a token id integer and encoded into bytes

  def save(self, file_prefix):
    """
    Saves two files: file_prefix.vocab and file_prefix.model. The model file is
    used to load the model using load(), but vocab file is just a pretty printed
    version to see
    """
    model_file = file_prefix + '.model'
    with open(model_file, 'w') as f:
      f.write('my_tokenizer_save\n')
      f.write(f"{self.pattern}\n")
      f.write(f"{len(self.special_tokens)}\n")
      for special, index in self.special_tokens.items():
        f.write(f'{special} {index}\n')
      for index1, index2 in self.merges:
        f.write(f'{index1} {index2}\n')

    vocab_file = file_prefix + '.vocab'
    inverted_merges = {index: pair for pair, index in self.merges.items()}
    with open(vocab_file, 'w', encoding='utf-8') as f:
      for index, token in self.vocab.items():
        render = render_token(token)
        if index in inverted_merges:
          index1, index2 = inverted_merges[index]
          render_index1 = render_token(self.vocab[index1])
          render_index2 = render_token(self.vocab[index2])
          f.write(f"[{render_index1} {render_index2}] -> [{render}] {index}\n")
        else:
          f.write(f"[{render}] {index}\n")
   # The save function contains two parts model file and svae file, the model
   # file is holding the essential parts of tokenizer like pattern, special token
   # which is seperated by the line by length of special tokens, ie, when giving
   # length of special tokens, it returns number of special token in top before
   # before printing special tokens and then writes the special tokens with its
   # index and at last it write the keys of merges dict,ie, index1,index, like
   # (97,98) are tuple as key in merge dict when iterating unpacks and write as
   # 97 98
   # The vocab file actually not used for traning purpose or loading it just used
   # for showing the vocab and can look into it and furthur modify. The inverted
   # merges is taken which is opposite of merge with index: pair dict(like {
   # (97,98): 256}) and when writing the file, iterating through vocab dict which
   # is index: pair like(256: b'a', b'b) beacuse vocab is index: bytes(index).
   # using render_token function decode it to its original string and do escape
   # control characters.Then the index(like 256,..) looks for match in inverted_
   # merges, if it is then the two varibles are got by goes through index of
   # inverted_merged and then unpacks the tuple. Then the each variable which
   # is unpacked tuple like(97,98) to 97, 98 goes to vocab dict and takes the
   # corresponding bytes like b'ab.. and then render using render_token function
   # At last writes as [a][b] -> [ab] 256. This is done to check that the char
   # which is formed from pair or to check children or leaf. The else check for
   # leaf as there is no match in index in inverted_merge dict and write as like
   # [a] - > 97

  def load(self, model_file):
    """ Load does the opposite of save(), it does
    only for model file
    """
    assert model_file.endswith('.model')
    merges = {}
    special_tokens = {}
    index = 256

    with open(model_file, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == 'my_tokenizer_save'
      self.pattern = f.readline().strip()
      num_special = int(f.readline().strip())
      for _ in range(num_special):
        special_token, special_index = f.readline().strip().split()
        special_tokens[special_token] = int(special_index)
      for line in f:
        index1, index2 = map(int, line.split())
        merges[(index1, index2)] = index
        index += 1

    self.merges = merges
    self.special_tokens = special_tokens
    self.vocab = self.build_vocab()
    # in load function checks for the model file which endswith
    # .model and initializing merges, special token dict and index
    # as 256. The model file is reading the lines it contains as first
    # it checks for the file the version as same as saved during save
    # function. Reads the pattern which is regex pattern basically used
    # in RegexTokenizer, then the number of special token saved during
    # save function, reads it and checks for the number of special tokens
    # and reads each by each by splitting and store it to dictionary
    # special tokens and then it reads for the remaining lines of merges
    # write during save function. The merges is saved in index1, index2
    # order so read it the order by splitting and stores as tuple of key
    # in merges dict. Like if 97,98 are the merge write into and it assigns
    # 256 the new idx to it and then adds by 1 in each iteration, atlast
    # the merges, special tokens are assigned to instances of Tokenizer class
    # and build the vocabulary