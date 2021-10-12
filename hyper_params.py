import os
from torch.cuda import is_available
# device HP
CUDA = is_available()
Device = 0

# experiment name HP
Language = 'en'
Paradigm = 'V'
ExpansionIterations = 0
OrthoAlg = False
ExpDir = ''     # should be set only for continued work on interrupted runs
FixLemmas = True

## PATHS
RootDir = os.path.dirname(os.path.abspath(__file__))
PreTrainedVecsPath = os.path.join(RootDir, 'vectors', '')
AllUnimorphPath = os.path.join(RootDir, 'unimorph', '')
ValidatedWordsPath = os.path.join(RootDir, 'words', '')
OutputsDir = os.path.join(RootDir, '')
if not os.path.isdir(OutputsDir): os.mkdir(OutputsDir)

# vocabulary HP
UseValidatedWords = True

# scoring HP
DeviationFrom = 'avg'       # can be either 'avg' or 'max'

# supervision enhancement HP
DeviationFromEnhance = 'max'       # can be either 'avg' or 'max'

# supervision HP
MaxSupervision = 5          # number of lemmas to include in the supervision set
MinPairsToConsider = 1      # minimal number of word pairs for a given label pair to include the label pair in the supervision
assert MinPairsToConsider < MaxSupervision

# misc HP
BatchSize = 20000
OptimalEdits = False if OrthoAlg else True
PlaceHolderToken = 'PLACE_HOLDER'

# surface version HP
OverlappedLetters = 2
LimitWordsForSurface = 200000

# write_dateset.py HP
TotExamples = 10000
CopyExamples = 1000
