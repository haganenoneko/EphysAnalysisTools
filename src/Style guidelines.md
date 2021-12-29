<!--
 Copyright (c) 2021 Delbert Yip
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

<style>
    pre {border: 5%; font-size: 14px;}
</style>

#  Style guidelines 
These are style guidelines (some are arbitrary) for personal use. They do not necessarily conform to PEP8 or other standards, but generally do. 

## Classes 
- every word in a class name is capitalized, e.g. `WordWordWord`
  - underscores are only used if: 
    - more than 3 words in the class name
    - two or more similar classes, e.g. `Word_A`, `Word_B`
- method names are in camel case, e.g. `wordWord`
  - leading underscore when the method is only called by one other function, i.e. a private function, e.g. `self._wordWord`
- class attributes names, e.g. `self._wordA_wordB_wordC`
  - prefixed by two underscores
  - fully lower case 
  - words are delimited by underscores
- class properties (`@property`) are like attributes, but are not prefixed by an underescore, e.g. `self.wordA_wordB_wordC`

## Architecture
### `@dataclass` for individual recordings
Each dataset consists of a CSV file and an ABF file. To hold these, we use the `AbstractRecording` `dataclass`. The `attrs` attribute is a dictionary that is intended to store arbitrary structures, such as:
- a `PdfPages` object to aggregate figures 
- leak-subtracted dataframe
- fitting results

### static methods group logically related methods
- `@staticmethod` is used abundantly to group methods that have a tight logical relationship
## Comments
- comments are sparse and/or one-liners, unless 
### Multiprocessing 
After files are selected and `AbstractRecording`s are instantiated for each selected file, 